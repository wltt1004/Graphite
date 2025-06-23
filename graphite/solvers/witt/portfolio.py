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

from typing import List, Union, Literal
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphV1PortfolioProblem, GraphV1PortfolioSynapse
from graphite.utils.graph_utils import timeout
from graphite.utils.graph_utils import timeout, get_portfolio_distribution_similarity
import numpy as np
import asyncio, tempfile
import time
import concurrent.futures
import random
import os, subprocess
import bittensor as bt

class portfolioSolver(BaseSolver):

    def __init__(self, problem_types:List[GraphV1PortfolioProblem]=[GraphV1PortfolioProblem()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int, offspring:int = 10)->List[int]:
        start_time = time.time()
        input_path = "../model/in.txt"
        output_path = "../model/out.txt"
        exe_path = "../model/portfolio"
        with open(input_path,'w') as f:
            f.write(f"{formatted_problem.n_portfolio} {len(formatted_problem.pools)}\n")
            for [tao, alpha] in formatted_problem.pools:
                f.write(f"{tao} {alpha}\n")
            for portfolio in initialPortfolios:
                for i in portfolio:
                    f.write(f"{i} ")
                f.write("\n")
            for i in constraintValues:
                f.write(f"{i} ")
            f.write("\n")
            for i in constraintTypes:
                f.write(f"{i} ")
            f.write("\n")

        result = subprocess.run([exe_path, input_path, output_path], capture_output=True, text=True)

        if result.returncode == 0:
            # Read the contents of the output file
            try:
                swaps = []
                with open(output_path, 'r') as f:
                    for line in f:
                        swap = line.strip().split()
                        swaps.append(swap)
                return swaps

            except FileNotFoundError:
                print(f"Error: The output file '{output_path}' was not found.")
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
        else:
            print(f"Error: C++ program failed with return code {result.returncode}")
            if result.stderr:
                print("C++ stderr:")
                print(result.stderr)


    def problem_transformations(self, problem: GraphV1PortfolioProblem):
        return problem
    
if __name__=="__main__":
    subtensor = bt.Subtensor("finney")
    subnets_info = subtensor.all_subnets()
    

    num_portfolio = random.randint(50, 200)
    pools = [[subnet_info.tao_in.rao, subnet_info.alpha_in.rao] for subnet_info in subnets_info]
    # print(pools)
    num_subnets = len(pools)
    avail_alphas = [subnet_info.alpha_out.rao for subnet_info in subnets_info]
    # print(num_portfolio, num_subnets)
    # for [tao, alpha] in pools:
    #     print(tao, alpha)
    # print(avail_alphas)
    # Create initialPortfolios: random non-negative token allocations
    initialPortfolios: List[List[int]] = []
    for _ in range(num_portfolio):
        portfolio = [int(random.uniform(0, avail_alpha//(2*num_portfolio))) if netuid != 0 else int(random.uniform(0, 10000*1e9/num_portfolio)) for netuid, avail_alpha in enumerate(avail_alphas)]  # up to 100k tao and random amounts of alpha_out tokens
        # On average, we assume users will invest in about 50% of the subnets
        portfolio = [portfolio[i] if random.random() < 0.5 else 0 for i in range(num_subnets)]
        initialPortfolios.append(portfolio)
    # for portfolio in initialPortfolios:
    #     for i in portfolio:
    #         print(i, end=' ')
    #     print("")
    # print(initialPortfolios)
    # print(pools[0][0], pools[0][1])

    # Create constraintTypes: mix of 'eq', 'ge', 'le'
    constraintTypes: List[str] = []
    for _ in range(num_subnets):
        constraintTypes.append(random.choice(["eq", "ge", "le", "ge", "le"])) # eq : ge : le = 1 : 2 : 2

    # Create constraintValues: match the types
    constraintValues: List[Union[float, int]] = []
    for ctype in constraintTypes:
        # ge 0 / le 100 = unconstrained subnet
        if ctype == "eq":
            constraintValues.append(random.uniform(0.5, 3.0))  # small fixed value
        elif ctype == "ge":
            constraintValues.append(random.uniform(0.0, 5.0))   # lower bound
        elif ctype == "le":
            constraintValues.append(random.uniform(10.0, 100.0))  # upper bound
    
    for idx, constraintValue in enumerate(constraintValues):
        if random.random() < 0.5:
            constraintTypes[idx] = "eq"
            constraintValues[idx] = 0

    ### Adjust constraintValues in-place to make sure feasibility is satisfied.
    eq_total = sum(val for typ, val in zip(constraintTypes, constraintValues) if typ == "eq")
    min_total = sum(val for typ, val in zip(constraintTypes, constraintValues) if typ in ("eq", "ge"))
    max_total = sum(val if typ in ("eq", "le") else 100 for typ, val in zip(constraintTypes, constraintValues))

    # If eq_total > 100, need to scale down eq constraints
    if eq_total > 100:
        scale = 100 / eq_total
        for i, typ in enumerate(constraintTypes):
            if typ == "eq":
                constraintValues[i] *= scale

    # After fixing eq, recompute min and max
    min_total = sum(val for typ, val in zip(constraintTypes, constraintValues) if typ in ("eq", "ge"))
    max_total = sum(val if typ in ("eq", "le") else 100 for typ, val in zip(constraintTypes, constraintValues))

    # If min_total > 100, reduce some "ge" constraints
    if min_total > 100:
        ge_indices = [i for i, typ in enumerate(constraintTypes) if typ == "ge"]
        excess = min_total - 100
        if ge_indices:
            for idx in ge_indices[::-1]:
                if constraintValues[idx] > 0:
                    reduction = min(constraintValues[idx], excess)
                    constraintValues[idx] -= reduction
                    excess -= reduction
                    if excess <= 0:
                        break

    # If max_total < 100, increase some "le" constraints
    if max_total < 100:
        le_indices = [i for i, typ in enumerate(constraintTypes) if typ == "le"]
        shortage = 100 - max_total
        if le_indices:
            for idx in le_indices:
                if constraintValues[idx] < 100:
                    increment = min(100-constraintValues[idx], shortage)
                    constraintValues[idx] += increment
                    shortage -= increment
                    if shortage <= 0:
                        break
    
    # Final clip: make sure no negatives
    for i in range(len(constraintValues)):
        constraintValues[i] = max(0, constraintValues[i])

    # for i in constraintValues:
    #     print(i, end=' ')
    # for i in constraintTypes:
    #     print(i, end=' ')
    test_problem = GraphV1PortfolioProblem(problem_type="PortfolioReallocation", 
                                    n_portfolio=num_portfolio, 
                                    initialPortfolios=initialPortfolios, 
                                    constraintValues=constraintValues,
                                    constraintTypes=constraintTypes,
                                    pools=pools)

    # print(constraintValues)
    # print(constraintTypes)
    solver1 = portfolioSolver(problem_types=[test_problem])

    async def main(timeout):
        try:
            route1 = await asyncio.wait_for(solver1.solve_problem(test_problem), timeout=timeout)
            return route1
        except asyncio.TimeoutError:
            print(f"Solver1 timed out after {timeout} seconds")
            return None  # Handle timeout case as needed

    start_time = time.time()
    swaps = asyncio.run(main(10)) 
    print("Swaps:", swaps, "\nNum portfolios:", test_problem.n_portfolio, "\nSubnets:", len(test_problem.constraintTypes), "\nTime Taken:", time.time()-start_time)

    # swaps = swaps[:-1]
    # print("initialPortfolios", test_problem.initialPortfolios)
    test_synapse = GraphV1PortfolioSynapse(problem = test_problem, solution = swaps)

    print("ok")
    swaps, objective_score = get_portfolio_distribution_similarity(test_synapse)
    print("Swaps:", swaps, "\nScore:", objective_score)