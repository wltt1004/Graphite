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
from graphite.protocol import GraphV1Problem, GraphV2Problem
from graphite.utils.graph_utils import timeout
import numpy as np
import asyncio, tempfile
import time
import concurrent.futures
import random
import os, subprocess

class tspSolver(BaseSolver):

    def __init__(self, problem_types:List[Union[GraphV1Problem, GraphV2Problem]]=[GraphV1Problem(n_nodes=2), GraphV1Problem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id:int, run = 1, pop = 100, offspring=30, timelimit=12)->List[int]:
        start_time = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            tsp_path = "../model/tsp.tsp"
            tour_path = "best"
            with open(tsp_path,'w') as f:
                f.write(f"NAME : tsp\n")
                f.write(f"COMMENT : Graphite tsp solver\n")
                f.write(f"TYPE : TSP\n")
                f.write(f"DIMENSION : {len(formatted_problem)}\n")
                f.write(f"EDGE_WEIGHT_TYPE : GEOM\n")
                f.write(f"NODE_COORD_SECTION\n")
                for i, [x,y] in enumerate(formatted_problem):
                    f.write(f"{i+1} {x:.2f} {y:.2f}\n")
                f.write(f"EOF")

            proc = await asyncio.create_subprocess_exec(
                '../model/jikken', f"{run}", f"{tour_path}", f"{pop}", f"{offspring}", f"{tsp_path}", f"{timelimit}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            try:
                # Wait for completion but enforce wall-clock budget
                await asyncio.wait_for(proc.wait(), timeout=timelimit + 1)
            except asyncio.TimeoutError:
                # Gracefully interrupt GA-EX → allow it to dump best tour so far
                proc.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timelimit + 1)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
            stdout, stderr = await proc.communicate()
        tour_path += "_BestSol"
        with open(tour_path) as f:
            inputs = f.readlines()
            tour = list(map(int, inputs[1].split()))
            tour.append(tour[0])
            return tour

    def problem_transformations(self, problem: Union[GraphV1Problem, GraphV2Problem]):
        return problem.nodes
    
if __name__=="__main__":
    from graphite.data.distance import geom_edges, man_2d_edges, euc_2d_edges
    loaded_datasets = {}
    with np.load('../../../dataset/Asia_MSB.npz') as f:
        loaded_datasets["Asia_MSB"] = np.array(f['data'])
    def recreate_nodes(problem: GraphV2Problem):
        node_coords_np = loaded_datasets["Asia_MSB"]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        return node_coords
    def recreate_edges(problem: GraphV2Problem):
        node_coords_np = loaded_datasets[problem.dataset_ref]
        node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
        if problem.cost_function == "Geom":
            return geom_edges(node_coords)
        elif problem.cost_function == "Euclidean2D":
            return euc_2d_edges(node_coords)
        elif problem.cost_function == "Manhatten2D":
            return man_2d_edges(node_coords)
        else:
            return "Only Geom, Euclidean2D, and Manhatten2D supported for now."
      
    n_nodes = random.randint(4500, 5000)
    # randomly select n_nodes indexes formatted_problemfrom the selected graph
    selected_node_idxs = random.sample(range(26000000), n_nodes)
    test_problem = GraphV2Problem(problem_type="Metric TSP", n_nodes=n_nodes, selected_ids=selected_node_idxs, cost_function="Geom", dataset_ref="Asia_MSB")
    if isinstance(test_problem, GraphV2Problem):
        test_problem.edges = recreate_edges(test_problem)
        test_problem.nodes = recreate_nodes(test_problem)
    # print("Problem", test_problem)
    solver = tspSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")

    solver = tspSolver(problem_types=[test_problem])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time}")