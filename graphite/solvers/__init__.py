from .base_solver import BaseSolver
from .beam_solver import BeamSearchSolver
from .exact_solver import DPSolver
from .greedy_solver import NearestNeighbourSolver
from .hpn_solver import HPNSolver
from .greedy_solver_multi import NearestNeighbourMultiSolver
from .greedy_solver_multi_2 import NearestNeighbourMultiSolver2
from .greedy_solver_multi_3 import NearestNeighbourMultiSolver3
from .greedy_solver_multi_4 import NearestNeighbourMultiSolver4
from .insertion_solver_multi import InsertionMultiSolver
from .greedy_portfolio_solver import GreedyPortfolioSolver

from .witt.mdmtsp import mdmtspSolver
from .witt.tsp import tspSolver
from .witt.cmdmtsp import cmdmtspSolver
from .witt.mtsp import mtspSolver
from .witt.portfolio import portfolioSolver