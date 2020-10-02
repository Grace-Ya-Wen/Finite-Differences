# import statements
from functions_part2 import *


# define the boundary and initial condition functions
def left(x, t): return 0.*x+0.*t
def right(x, t): return 0.*x+0.*t
def initial(x, t): return np.sin(np.pi*x)+0.*t


# define the boundaries properties and create each BoundariesXY object
x0 = Boundary('dirichlet', left)
x1 = Boundary('dirichlet', right)
t0 = Boundary('initial', initial)

# create an instance of the MeshXY class for each solution method
mesh_explicit = MeshXT(x=[0., 1.], t=[0., 0.2], delta=[0.05, 0.001], boundaries=[x0, x1, t0])
mesh_implicit = MeshXT(x=[0., 1.], t=[0., 0.2], delta=[0.05, 0.001], boundaries=[x0, x1, t0])

# create an instance of the SolverHeatXT class for each solution method
solver_explicit = SolverHeatXT(mesh_explicit, alpha=1., method='explicit')
solver_implicit = SolverHeatXT(mesh_implicit, alpha=1., method='crank-nicolson')

# run the solver for each case
solver_explicit.solver(mesh_explicit)
solver_implicit.solver(mesh_implicit)

# plot the solution to the PDE on the mesh
mesh_explicit.plot_solution(ntimes=5, save_to_file=False, save_file_name='task4_explicit.png')
mesh_implicit.plot_solution(ntimes=5, save_to_file=False, save_file_name='task4_implicit.png')
