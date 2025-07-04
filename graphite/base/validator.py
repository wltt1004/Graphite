# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import copy
import numpy as np
import asyncio
import argparse
import threading
import bittensor as bt
import time
import random
import math

from abc import abstractmethod
from typing import List, Union
from traceback import print_exception
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum

from graphite.base.neuron import BaseNeuron
from graphite.base.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit #TODO: Replace when bittensor switches to numpy
from graphite.mock import MockDendrite
from graphite.utils.config import add_validator_args
from graphite.utils.yield_utils import fetch_rebalancing_axon, fetch_performance_axon
from graphite.protocol import IsAlive
from graphite.utils.uids import check_uid_availability

import requests

from dotenv import load_dotenv
import os
import wandb
import shutil
import logging

load_dotenv()

class ScoreType(Enum):
    ORGANIC = "organic"
    SYNTHETIC = "synthetic"
    YIELD = "yield"

class CompositeScore(BaseModel):
    organic_score: np.ndarray = Field(default_factory=list)
    synthetic_score: np.ndarray = Field(default_factory=list)
    yield_score: np.ndarray = Field(default_factory=list)

    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True
        )

    @property
    def composite_score(self):
        '''
        Returns net score for each uid with the weight distribution:
        organic_score: 0.2
        synthetic_score: 0.6
        yield_score: 0.2
        '''
        yield_score_ = np.array([0 if y is None or (isinstance(y, float) and np.isnan(y)) else y for y in self.yield_score])
        score = self.organic_score * 0.2 + self.synthetic_score * 0.6 + yield_score_ * 0.2
        mask = np.array([np.isnan(y) for y in self.yield_score])
        score[mask] = 0

        return score
    

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        add_validator_args(cls, parser)

    @abstractmethod
    def organic_forward(self):
        pass

    @abstractmethod
    def yield_forward(self):
        pass

    def __init__(self, config=None):
        super().__init__(config=config)

        # Save a copy of the hotkeys to local memory.
        ## |__ Moved to load_state() which implicitly sets hotkeys
        self.load_state()
        # instantiate wandb
        self.instantiate_wandb()

        # Dendrite lets us send messages to other nodes (axons) in the network.
        if self.config.mock:
            self.dendrite = MockDendrite(wallet=self.wallet)
        else:
            self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        
        bt.logging.info(f"Initiating validator with scores: {self.scores}")

        self.uid_query_sets = []

        self.organic_axon = fetch_rebalancing_axon() # gets the axon info of the neuron that serves the rebalancing data
        self.yield_axon = fetch_performance_axon() # gets the axon info of the neuron that serves the performance data
        bt.logging.info(f"Rebalancing axon: {self.organic_axon}")
        bt.logging.info(f"Performance axon: {self.yield_axon}")
        
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Harded coded for now until new API release
        self.bearer_token_is_valid = False

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()
        

    async def get_available_uids(self):
        available_uids = {}

        for uid in range(self.metagraph.n.item()):
            uid_is_available = check_uid_availability(
                self.metagraph, uid, self.config.neuron.vpermit_tao_limit
            )

            if uid_is_available:
                available_uids[uid] = uid

        return available_uids
    
    async def get_miner_uids(self):
        # using the inverse of the validator recognition logic in blacklist fn
        # Check if uid has validator permit
        def valid_ip(ip: str):
            return not "0.0.0.0" in ip
        miner_uids = []
        for uid in self.metagraph.uids:
            if self.metagraph.validator_permit[uid]:
                # this uid is recognized as a validator
                continue
            elif self.metagraph.stake[uid] >= 2000:
                # this amount of stake is too high for a miner
                continue
            elif not valid_ip(self.metagraph.axons[uid].ip):
                continue
            else:
                miner_uids.append(uid)
        return miner_uids

    
    async def get_available_uids_alive(self):
        # get all axons in subnet
        all_axons = self.metagraph.axons

        # checks IsAlive() for each uid
        tasks = [self.check_alive(axon, all_axons.index(axon)) for axon in all_axons]
        results = await asyncio.gather(*tasks)
       
        # dictionary that maps UID -> AxonInfo for all alive UIDs
        available_uids = {uid: axon for uid, axon in enumerate(results) if axon is not None}

        return available_uids
    
    async def get_k_uids(self, k:int=30):
        if len(self.uid_query_sets) > 0:
            available_uids = await self.get_available_uids()
            to_query = {}
            for uid in self.uid_query_sets[0]:
                if uid in available_uids:
                    to_query[uid] = available_uids[uid]
            self.uid_query_sets.pop(0)
            return to_query
        else:
            available_uids = await self.get_available_uids()
            random_uids = random.sample(list(available_uids.keys()), min(k, len(available_uids)))
            incentives = self.metagraph.I
            incentive_indexed = {key + index/10000000 + random.random()/1000000: index for index, key in enumerate(incentives)}
            incentives_ranked = [incentive_indexed[key] for key in sorted(incentive_indexed.keys())]
            incentives_ranked_final = [i for i in incentives_ranked if i in available_uids.keys()]
            group_size = len(incentives_ranked_final) // k
            excess = len(incentives_ranked_final) - k * group_size
            groups = [incentives_ranked_final[i * group_size:(i + 1) * group_size] for i in range(k - excess)] 
            incentives_ranked_final_rem = incentives_ranked_final[(k - excess)*group_size:]
            groups2 = [incentives_ranked_final_rem[i * (group_size+1):(i + 1) * (group_size+1)] for i in range(excess)]
            groups += groups2
            num_groups = math.ceil(len(incentives_ranked_final) / k)
                
            query_sets = []
            for _ in range(num_groups):  
                current_selection = []
                for group in groups:
                    if len(group) > 1:
                        selected = random.choice(group)
                        group.remove(selected)  
                        current_selection.append(selected)
                    else:
                        current_selection.append(group[0])
                query_sets.append(current_selection)
            self.uid_query_sets = query_sets

            if len(self.uid_query_sets) > 0:
                to_query = {}
                for uid in self.uid_query_sets[0]:
                    if uid in available_uids:
                        to_query[uid] = available_uids[uid]
                self.uid_query_sets.pop(0)
                return to_query
            else:
                available_uids = await self.get_available_uids()
                random_uids = random.sample(list(available_uids.keys()), min(k, len(available_uids)))
                return {uid: available_uids[uid] for uid in random_uids}

    async def get_top_k_uids(self, k:int=30, alpha:float=0.7):
        assert (alpha<=1) and (alpha>0.5), ValueError("For the get_top_k_uids method, alpha needs to be between 0.5 and 1")
        # get available_uids
        available_uids = await self.get_available_uids_alive()
        incentives = self.metagraph.I
        available_uids_and_incentives = [(uid, incentives[uid]) for uid in available_uids.keys()]
        sorted_axon_list = sorted(available_uids_and_incentives, key=lambda x: x[1], reverse=True)
        # query a random sample of half of the top 10% of miners:
        top_k_axons = sorted_axon_list[:min(len(sorted_axon_list),k)]
        if len(sorted_axon_list) > k:
            bottom_remainder = math.floor(k*(1-alpha))
            if bottom_remainder > (len(sorted_axon_list)-k):
                bottom_remainder = len(sorted_axon_list) - k
            top_n = k - bottom_remainder
            assert (top_n>0) and (bottom_remainder>=0), ValueError(f'Invalid call values: calling {top_n} top miners and {bottom_remainder} bottom miners')
            other_axons = [x[0] for x in random.sample(sorted_axon_list[k:], bottom_remainder)]
            random_top_axons = [x[0] for x in random.sample(top_k_axons, top_n)]
            selected_uids = random_top_axons + other_axons
            return {uid: available_uids[uid] for uid in selected_uids}
        else:
            return {uid: available_uids[uid] for uid, incentive in sorted_axon_list}

    async def check_alive(self, axon, uid):
        # check if axon is alive
        try:
            response = await self.dendrite(axon, IsAlive(), deserialize=False, timeout=30)
            if response.is_success:
                # bt.logging.info(f"UID {uid} is alive")
                # bt.logging.info("Response: ", response)
                hotkey = axon.hotkey
                dend_hotkey = self.wallet.hotkey.ss58_address
                log_line = f"{hotkey[:5]}_{dend_hotkey[:5]}_{time.time()}\n"
                with open("is_alive_logs.txt", "a") as f:
                    f.write(log_line)
                return axon
            
            # bt.logging.info(f"UID {uid} is not alive")
            return None
        
        except Exception as e:
            bt.logging.error(f"Error when checking UID is_alive: {e}")
            return None

    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(
                f"Failed to create Axon initialize with exception: {e}"
            )
            pass
    
    async def stagger_forward(self):
        await asyncio.sleep(1)
        return await self.forward() # return the co-routine

    async def concurrent_forward(self):
        
        num_concurrent_forwards = self.config.neuron.num_concurrent_forwards

        self.current_num_concurrent_forwards = num_concurrent_forwards
        bt.logging.info(f"Concurrent forwards: {num_concurrent_forwards}")

        coroutines = [
            self.stagger_forward()
            for _ in range(num_concurrent_forwards) # self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def instantiate_wandb(self):
        load_dotenv()
        # self.organic_endpoint = os.getenv('MONGODB_ENDPOINT')
        # self.db_bearer_token = os.getenv('MONGODB_BEARER_TOKEN')
        wandb_api_key = os.getenv('WANDB_API_KEY')
        
        if not wandb_api_key:
            raise ValueError("wandb_api_key is not set in the .env file.")

        os.environ["WANDB__SERVICE_WAIT"]="300"
        os.environ["WANDB_SILENT"] = "true"                     
        os.environ["wandb_key"]=wandb_api_key
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.ERROR)

        wandb.login(key=os.environ["wandb_key"])
        bt.logging.info(f"Logged in with key {wandb_api_key}")

    def cleanup_wandb(self, wandb):
        try:
            run_dir = wandb.run.dir
            wandb.finish()
            if os.path.exists(run_dir):
                parent_dir = os.path.dirname(run_dir)
                shutil.rmtree(parent_dir)
                bt.logging.debug(f"Deleted {run_dir}")
        except:
            pass

    async def organic_request_flow(self):
        # trigger organic forward flow
        while True:
            try:
                bt.logging.info(f"Organic request flow")
                if self.organic_axon is None:
                    bt.logging.error(f"Rebalancing axon is not set")
                    self.organic_axon = fetch_rebalancing_axon()
                else:
                    await self.organic_forward()
            except KeyboardInterrupt:
                bt.logging.info(f"Organic request flow interrupted by keyboard interrupt")
                break
            except Exception as e:
                bt.logging.error(f"Error in organic request flow: {e}")
                continue
            finally:
                # poll for a new organmic rebalance job every block
                await asyncio.sleep(12)
            # Let the step management be handled by the synthetic request flow

    async def yield_request_flow(self):
        while True:
            try:
                bt.logging.info(f"Yield request flow")
                if self.yield_axon is None:
                    bt.logging.error(f"Yield axon is not set")
                    self.yield_axon = fetch_performance_axon()
                else:
                    await self.yield_forward()
            except KeyboardInterrupt:
                bt.logging.info(f"Yield request flow interrupted by keyboard interrupt")
                break
            except Exception as e:
                bt.logging.error(f"Error in yield request flow: {e}")
                continue
            finally:
                await asyncio.sleep(12)

    async def synthetic_request_flow(self):
        while True:
            try:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                if self.block - self.last_synthetic_req > 25: # synthetic request every 25 blocks ~ 5 min
                    self.last_synthetic_req = self.block
                    self.concurrencyIdx = 0
                    await self.concurrent_forward()

                # re-align scoring after X validator epochs 
                if self.block - self.last_load_state > 360*3:
                    self.last_load_state = self.block
                    self.load_state()

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and set weights if 1 epoch (300 blocks) has passed.
                bt.logging.info(f"METAGRAPH UPDATE {time.time()}")
                self.sync()

                # increment step each block and update view on the metagraph
                self.step += 1
            except KeyboardInterrupt:
                bt.logging.info(f"Synthetic request flow interrupted by keyboard interrupt")
                break
            except Exception as e:
                bt.logging.error(f"Error in synthetic request flow: {e}")
                continue
            await asyncio.sleep(12)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")
        self.last_synthetic_req = 0
        self.last_load_state = 0

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            # Create tasks for both request flows
            self.organic_task = self.loop.create_task(self.organic_request_flow())
            self.synthetic_task = self.loop.create_task(self.synthetic_request_flow())
            self.yield_task = self.loop.create_task(self.yield_request_flow())

            # Run the event loop
            self.loop.run_forever()

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            self.loop.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error(f"Error during validation: {str(err)}")
            bt.logging.debug(
                str(print_exception(type(err), err, err.__traceback__))
            )

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            # Cancel the running tasks
            if hasattr(self, 'organic_task'):
                self.organic_task.cancel()
            if hasattr(self, 'synthetic_task'):
                self.synthetic_task.cancel()
            if hasattr(self, 'yield_task'):
                self.yield_task.cancel()
            self.loop.stop()
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()

    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores.composite_score).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        try:
            raw_weights = self.scores.composite_score / np.linalg.norm(self.scores.composite_score, ord=1, axis=0, keepdims=True)
        except RuntimeWarning:
            bt.logging.info("Sum of scores = 0, skip setting weights for now")
            return None

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)


        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return
        
        # Fetch the yield and organic axons
        try:
            self.yield_axon = fetch_performance_axon()
        except Exception as e:
            bt.logging.error(f"Error fetching yield axon during resync: {e}")
        try:
            self.organic_axon = fetch_rebalancing_axon()
        except Exception as e:
            bt.logging.error(f"Error fetching organic axon during resync: {e}")
        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            for score_type in ScoreType:
                # Update the size of the moving average scores.
                new_moving_average = np.zeros((self.metagraph.n))
                min_len = min(len(self.hotkeys), len(self.scores))
                new_moving_average[:min_len] = getattr(self.scores, score_type.value)[:min_len]
                setattr(self.scores, score_type.value, new_moving_average)

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int], score_type: ScoreType):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""

        scoring_suffix = "_score"
        # Check if rewards contains NaN values.

        rewards_dup = rewards.copy()

        if np.isnan(rewards).any():
            # bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            # Replace any NaN values in rewards with 0.
            rewards = np.nan_to_num(rewards, nan=0)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores.composite_score)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {rewards}")

        # Update scores with rewards produced by this step.
        # shape: [ metagraph.n ]
        alpha: float = self.config.neuron.moving_average_alpha
        current_score = getattr(self.scores, score_type.value + scoring_suffix)
        current_score = alpha * scattered_rewards + (
                1 - alpha
            ) * current_score
        
        if score_type == ScoreType.YIELD:
            for idx, uid in enumerate(uids_array):
                if np.isnan(rewards_dup[idx]):
                    current_score[uid] = np.nan

        setattr(self.scores, score_type.value + scoring_suffix, current_score)
        bt.logging.debug(f"Updated moving avg scores: {self.scores}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores.model_dump(),
            hotkeys=self.hotkeys,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")

        # Load the state of the validator from file. --> Check if state exists if not initiate using default values

        # if os.path.exists(self.config.neuron.full_path + "/state.npz"):
        #     # load in state
        #     try:
        #         state = np.load(self.config.neuron.full_path + "/state.npz")
        #         self.step = state["step"]
        #         self.scores = state["scores"]
        #         self.hotkeys = state["hotkeys"]
        #     except EOFError:
        #         current_incentive = np.array(self.metagraph.I)
        #         self.scores = (current_incentive - np.min(current_incentive))/(np.max(current_incentive)-np.min(current_incentive))
        #         self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # else:
            # self.step = 0 # already set in BaseNeuron init
        current_incentive = np.array(self.metagraph.I)
        if np.max(current_incentive) - np.min(current_incentive) != 0:
            metagraph_scores = (current_incentive - np.min(current_incentive))/(np.max(current_incentive)-np.min(current_incentive))
        else:
            metagraph_scores = np.zeros_like(current_incentive)


        self.scores = CompositeScore(organic_score=metagraph_scores, synthetic_score=metagraph_scores, yield_score=metagraph_scores)

        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        
        
        # call resync metagraph to make appropriate updates from saved state
        self.resync_metagraph()