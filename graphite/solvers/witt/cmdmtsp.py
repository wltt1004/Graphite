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
from graphite.protocol import GraphV1Problem, GraphV2Problem, GraphV2ProblemMulti, GraphV2ProblemMultiConstrained, GraphV2Synapse
from graphite.utils.graph_utils import timeout, get_multi_minmax_tour_distance
from graphite.data.dataset_utils import load_default_dataset
from graphite.data.distance import geom_edges, euc_2d_edges, man_2d_edges
import numpy as np
import time
import asyncio
import random
import math
import bittensor as bt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import sys

class cmdmtspSolver(BaseSolver):
    '''
    This solver is a constructive nearest_neighbour algorithm that assigns cities to subtours based on the min increase in objective function value.
    '''
    def __init__(self, problem_types:List[GraphV2Problem]=[GraphV2ProblemMultiConstrained()]):
        super().__init__(problem_types=problem_types)
    
    def solve_with_bound(self, matrix, demands, capacities, starts, ends, dist_bound: int, params):
        n = len(matrix)
        k = len(starts)
        print(type(dist_bound))
        mgr = pywrapcp.RoutingIndexManager(n, k, starts, ends)
        routing = pywrapcp.RoutingModel(mgr)

        # Distance callback
        def dist_cb(i, j):
            return matrix[mgr.IndexToNode(i)][mgr.IndexToNode(j)]

        dist_idx = routing.RegisterTransitCallback(dist_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(dist_idx)

        # Distance dimension (for min‑max objective)
        routing.AddDimension(dist_idx, 0, dist_bound, True, "Distance")
        dist_dim = routing.GetDimensionOrDie("Distance")
        dist_dim.SetGlobalSpanCostCoefficient(1)  # minimises max tour

        # Capacity dimension (if any demand > 0)
        if any(demands):
            def demand_cb(i):
                return demands[mgr.IndexToNode(i)]

            dem_idx = routing.RegisterUnaryTransitCallback(demand_cb)
            routing.AddDimensionWithVehicleCapacity(dem_idx, 0, capacities, True, "Capacity")

        # Solve
        solution = routing.SolveWithParameters(params)
        return routing, mgr, dist_dim, solution

    async def solve(self, formatted_problem, future_id:int)->List[int]:
        starts = formatted_problem.depots
        ends = starts
        n_nodes = formatted_problem.n_nodes
        n_salesmen = formatted_problem.n_salesmen
        print(n_salesmen)
        # Load distance matrix
        matrix_ = formatted_problem.edges * 1000
        matrix = matrix_.astype(int)
        demands = formatted_problem.demand
        capacities = formatted_problem.constraint
        print(len(matrix[0]))
        print(capacities)
        print(starts)
        print(ends)
        loose_bound = sum(max(row) for row in matrix)
        print(loose_bound)
        lo_params = pywrapcp.DefaultRoutingSearchParameters()
        pywrapcp.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        pywrapcp.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
        )
        lo_params.time_limit.seconds = 30
        r1, m1, d1, s1 = self.solve_with_bound(matrix, demands, capacities, starts, ends, int(loose_bound), lo_params)
        if not s1:
            print("ERROR: no solution in Phase 2.")

        # Print routes and distances
        span_max = 0
        routes = []
        for v in range(formatted_problem.n_salesmen):
            idx = r1.Start(v)
            route_nodes = []
            load = 0
            while not r1.IsEnd(idx):
                node = m1.IndexToNode(idx)
                # route_nodes.append(node)
                routes.append(node)
                load += demands[node]
                idx = s1.Value(r1.NextVar(idx))
            routes.append(m1.IndexToNode(idx))
            # route_nodes.append(m1.IndexToNode(idx))  # end depot
            # routes.append(route_nodes)
            span = s1.Value(d1.CumulVar(r1.End(v)))
            span_max = max(span_max, span)
            print(f"Vehicle {v}: dist={span}, load={load}, route={route_nodes}")
        print(span_max)
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
        def recreate_nodes(self, problem: Union[GraphV2Problem, GraphV2ProblemMulti]):
            node_coords_np = self.loaded_datasets[problem.dataset_ref]["data"]
            node_coords = np.array([node_coords_np[i][1:] for i in problem.selected_ids])
            return node_coords
    mock = Mock()
    load_default_dataset(mock)

    n_nodes = 500
    m = 5

    constraint = []
    # depots = sorted(random.sample(list(range(n_nodes)), k=m))
    depots = [11, 32, 55, 54, 78]

    demand = [1] * n_nodes
    for depot in depots:
        demand[depot] = 0
    total_demand_padded = sum(demand) + 9*m # padded to prevent invalid knap-sack problem conditions
    constraint = [(math.ceil(total_demand_padded/m) + random.randint(0, int(total_demand_padded/m * 0.3)) - random.randint(0, int(total_demand_padded/m * 0.2))) for _ in range(m-1)]
    constraint += [(math.ceil(total_demand_padded/m) + random.randint(0, int(total_demand_padded/m * 0.3)) - random.randint(0, int(total_demand_padded/m * 0.2)))] if sum(constraint) > total_demand_padded - (math.ceil(total_demand_padded/m) - random.randint(0, int(total_demand_padded/m * 0.2))) else [(total_demand_padded - sum(constraint) + random.randint(int(total_demand_padded/m * 0.2), int(total_demand_padded/m * 0.3)))]
    
    visitable_nodes = n_nodes - m
    # for i in range(m-1):
    #     # ensure each depot should at least have 2 nodes to visit
    #     constraint.append(random.randint(2, visitable_nodes-sum(constraint)-2*(m-i-1)))
    # constraint.append(visitable_nodes - sum(constraint))
    # # add a layer of flexibility
    # constraint = [con+random.randint(0, 100) for con in constraint]

    # constraint = [(math.ceil(n_nodes/m) + random.randint(0, int(n_nodes/m * 0.3)) - random.randint(0, int(n_nodes/m * 0.2))) for _ in range(m-1)]
    # constraint += [(math.ceil(n_nodes/m) + random.randint(0, int(n_nodes/m * 0.3)) - random.randint(0, int(n_nodes/m * 0.2)))] if sum(constraint) > n_nodes - (math.ceil(n_nodes/m) - random.randint(0, int(n_nodes/m * 0.2))) else [(n_nodes - sum(constraint) + random.randint(int(n_nodes/m * 0.2), int(n_nodes/m * 0.3)))]
    # constraint = [114, 104, 96, 120, 123]
    
    test_problem = GraphV2ProblemMultiConstrained(problem_type="Metric cmTSP", 
                                            n_nodes=n_nodes, 
                                            selected_ids=random.sample(list(range(100000)),n_nodes), 
                                            cost_function="Geom", 
                                            dataset_ref="Asia_MSB", 
                                            n_salesmen=m, 
                                            depots=depots, 
                                            single_depot=False,
                                            demand=demand,
                                            constraint=constraint)
    test_problem.edges = mock.recreate_edges(test_problem)
    test_problem.nodes = mock.recreate_nodes(test_problem)
    print("Depots", depots)
    print("Constraints", constraint, sum(constraint), sum(demand), sum(constraint) >= sum(demand))
    print("Demand", demand)
    
    print("Running NNMS4")
    solver1 = cmdmtspSolver(problem_types=[test_problem])
    start_time = time.time()
    # route1 = asyncio.run(solver1.solve_problem(test_problem))

    async def main(timeout):
        try:
            route1 = await asyncio.wait_for(solver1.solve_problem(test_problem), timeout=timeout)
            return route1
        except asyncio.TimeoutError:
            print(f"Solver1 timed out after {timeout} seconds")
            return None  # Handle timeout case as needed

    route1 = asyncio.run(main(10)) 
    print(route1)

    # test_synapse = GraphV2Synapse(problem = test_problem, solution = route1)
    # score1 = get_multi_minmax_tour_distance(test_synapse)

    # print(f"{solver1.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time()-start_time} and Salesmen: {m}")
    # print(f"Multi2 scored: {score1}")
    # print("Length of outes", [len(route) for route in route1])
    # print("Routes:")
    # [print(route) for route in route1]
    # route_demands = [[demand[idx] for idx in route] for route in route1]
    # # print("Double check tour demands:", route_demands)
    # print("Double check tour demands:", [sum(route_demand) for route_demand in route_demands])
    