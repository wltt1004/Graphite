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

from typing import List, Union
import matplotlib.pyplot as plt
from graphite.solvers.base_solver import BaseSolver
from graphite.solvers.greedy_solver_multi import NearestNeighbourMultiSolver
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2Synapse
from graphite.utils.graph_utils import timeout, get_multi_minmax_tour_distance
from graphite.data.dataset_utils import load_default_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
import multiprocessing
import numpy as np
import time
import asyncio
import random
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import bittensor as bt
import sys

class mdmtspSolver(BaseSolver):
    '''
    This solver is a constructive nearest_neighbour algorithm that assigns cities to subtours based on the min increase in objective function value.
    '''
    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2ProblemMulti()]):
        super().__init__(problem_types=problem_types)
    
    def solve_with_bound(self, matrix, starts, ends, bound: int, params):
        """
        Build and solve MDMTSP with given distance bound and search parameters.
        Returns (routing, manager, dimension, solution).
        """
        n = len(matrix)
        k = len(starts)
        manager = pywrapcp.RoutingIndexManager(n, k, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def dist_cb(from_idx, to_idx):
            return matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]

        transit_idx = routing.RegisterTransitCallback(dist_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Distance dimension with upper bound = bound
        routing.AddDimension(transit_idx, 0, bound, True, 'Dist')
        dist_dim = routing.GetDimensionOrDie('Dist')
        dist_dim.SetGlobalSpanCostCoefficient(1)

        # Solve
        solution = routing.SolveWithParameters(params)
        return routing, manager, dist_dim, solution
    
    async def solve(self, formatted_problem, future_id:int)->List[int]:
        starts = formatted_problem.depots
        ends = starts
        n_nodes = formatted_problem.n_nodes
        n_salesmen = formatted_problem.n_salesmen
        print(n_salesmen)
        # Load distance matrix
        matrix_ = formatted_problem.edges * 1000
        matrix = matrix_.astype(int)
        # print(matrix)
        # PHASE 1: quick bound discovery
        loose_bound = sum(max(row) for row in matrix)
        lo_params = pywrapcp.DefaultRoutingSearchParameters()
        lo_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        lo_params.time_limit.seconds = 2
        print(n_salesmen)
        print(starts)
        print(f"Phase 1: solving with loose_bound={loose_bound} (5 s)")
        r1, m1, d1, s1 = self.solve_with_bound(matrix, starts, ends, int(loose_bound), lo_params)
        if not s1:
            print("ERROR: no feasible solution in Phase 1.")
            sys.exit(1)

        # Extract max‑route length from solution
        bound = max(
            s1.Value(d1.CumulVar(r1.End(v)))
            for v in range(n_salesmen)
        )
        print(f"Discovered tight bound = {bound}")

        # PHASE 2: high‑quality solve under tight bound
        st = 0
        en = bound
        for t in range(6):
            mi = int((st + en) / 2)
            print(f"interval [{st}, {en}]\n")
            hi_params = pywrapcp.DefaultRoutingSearchParameters()
            hi_params.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            )
            hi_params.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
            )
            hi_params.time_limit.seconds = 2
            r2, m2, d2, s2 = self.solve_with_bound(matrix, starts, ends, mi, hi_params)
            if not s2:
                st = mi
            else:
                mx = 0
                for v in range(n_salesmen):
                    span = s2.Value(d2.CumulVar(r2.End(v)))
                    mx = max(mx, span)
                en = mx
        hi_params = pywrapcp.DefaultRoutingSearchParameters()
        hi_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        hi_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
        )
        hi_params.time_limit.seconds = 5
        r2, m2, d2, s2 = self.solve_with_bound(matrix, starts, ends, en, hi_params)

        if not s2:
            print("ERROR: no solution in Phase 2.")
            sys.exit(1)

        # Print routes and distances
        routes = []
        mx = 0
        for v in range(n_salesmen):
            idx = r2.Start(v)
            while not r2.IsEnd(idx):
                routes.append(m2.IndexToNode(idx))
                idx = s2.Value(r2.NextVar(idx))
            routes.append(m2.IndexToNode(idx))
            span = s2.Value(d2.CumulVar(r2.End(v)))
            routes.append(route)
            mx = max(mx, span)
        print(mx)
        return routes

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
            
    mock = Mock()
    load_default_dataset(mock)

    n_nodes = 2000
    m = 10

    depots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_problem = GraphV2ProblemMulti(n_nodes=n_nodes, single_depot=False, selected_ids=random.sample(list(range(100000)),n_nodes), dataset_ref="Asia_MSB", n_salesmen=m, depots=depots)
    test_problem.edges = mock.recreate_edges(test_problem)
    solver1 = mdmtspSolver(problem_types=[test_problem])
    start_time = time.time()
    route1 = asyncio.run(solver1.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route1)
    score1 = get_multi_minmax_tour_distance(test_synapse)
    solver2 = mdmtspSolver(problem_types=[test_problem])
    route2 = asyncio.run(solver2.solve_problem(test_problem))
    test_synapse = GraphV2Synapse(problem = test_problem, solution = route2)
    score2 = get_multi_minmax_tour_distance(test_synapse)
    # print(f"{solver.__class__.__name__} Solution: {route1}")
    print(f"{solver1.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time} and Salesmen: {m}")
    print(f"Multi2 scored: {score1} while Multi scored: {score2}")