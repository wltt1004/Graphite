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
from graphite.data.dataset_utils import load_default_dataset
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.utils.graph_utils import timeout
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
import numpy as np
import asyncio, tempfile
import time
import concurrent.futures
import random
import os, subprocess

class mtspSolver(BaseSolver):

    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int, run = 1, timelimit=15)->List[int]:
        start_time = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            par_path = "../model/mtsp.par"
            tsp_path = "mtsp.tsp"
            tour_path = "../model/best"
            exe_path = "../model/LKH"
            with open(tsp_path,'w') as f:
                f.write(f"NAME : mtsp\n")
                f.write(f"COMMENT : Graphite mtsp solver\n")
                f.write(f"TYPE : TSP\n")
                f.write(f"DIMENSION : {formatted_problem.n_nodes}\n")
                f.write(f"EDGE_WEIGHT_TYPE : GEOM\n")
                f.write(f"NODE_COORD_SECTION\n")
                for i, [x,y] in enumerate(formatted_problem.nodes):
                    f.write(f"{i+1} {x:.2f} {y:.2f}\n")
                f.write(f"EOF")
            with open(par_path, 'w') as f:
                f.write(f"SPECIAL\n")
                f.write(f"PROBLEM_FILE = mtsp.tsp\n")
                f.write(f"MAX_CANDIDATES = 3 SYMMETRIC\n")
                f.write(f"MOVE_TYPE = 5\n")
                f.write(f"RECOMBINATION = CLARIST\n")
                f.write(f"MAX_TRIALS = 1000000\n")
                f.write(f"RUNS = {run}\n")
                f.write(f"SEED = 0\n")
                f.write(f"SALESMEN = {formatted_problem.n_salesmen}\n")
                f.write(f"MTSP_MIN_SIZE = 0\n")
                f.write(f"MTSP_OBJECTIVE = MINMAX\n")
                f.write(f"MTSP_SOLUTION_FILE = {tour_path}\n")
                f.write(f"TIME_LIMIT = {timelimit}")
            
            proc = await asyncio.create_subprocess_exec(
                exe_path, "../model/mtsp.par",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            try:
                # Wait for completion but enforce wall-clock budget
                await asyncio.wait_for(proc.wait(), timeout=timelimit + 3)
            except asyncio.TimeoutError:
                # Gracefully interrupt GA-EX → allow it to dump best tour so far
                proc.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timelimit + 3)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            stdout, stderr = await proc.communicate()
        tours = []
        with open(tour_path) as f:
            for line in f:
                tour = line.strip().split()
                tour1 = list(map(int, tour))
                new_tour = [i - 1 for i in tour1]
                for i in new_tour:
                    tours.append(i)
        return tours

    def problem_transformations(self, problem: Union[GraphV2ProblemMulti]):
        return problem
    

if __name__=="__main__":
    # runs the solver on a test Metric mTSP
    class Mock:
        def __init__(self) -> None:
            pass

        def recreate_edges(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
            node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
            node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
            if problem.cost_function == "Geom":
                return geom_edges(node_coords)
            elif problem.cost_function == "Euclidean2D":
                return euc_2d_edges(node_coords)
            elif problem.cost_function == "Manhatten2D":
                return man_2d_edges(node_coords)
            else:
                return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
        def recreate_nodes(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
            node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
            node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
            return node_coords
    
    mock = Mock()

    load_default_dataset(mock)

    n_nodes = 600
    m = 5

    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, selected_ids=random.sample(list(range(100000)),n_nodes), dataset_ref="Asia_MSB", n_salesmen=m, depots=[0]*m)
    test_problem.edges = mock.recreate_edges(test_problem)
    test_problem.nodes = mock.recreate_nodes(test_problem)
    print(f"{n_nodes} {m}")
    
    solver = mtspSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")