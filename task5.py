# import statements
from functions_part2 import *


# define the boundary and initial condition functions
def left(x, t): return 0.*x+0.*t
def right(x, t): return 0.*x+0.*t
def initial(x, t): return np.sin(np.pi*x)+0.*t


# define the analytic solution function
def exact(x, t): return np.sin(np.pi*x)*np.exp(-1.*np.pi*np.pi*t)


# define the boundaries properties and create each BoundariesXY object
x0 = Boundary('dirichlet', left)
x1 = Boundary('dirichlet', right)
t0 = Boundary('initial', initial)

# create a mesh object for each case
mesh_explicit1 = MeshXT(x=[0., 1.], t=[0., 0.1], delta=[0.05, 0.001], boundaries=[x0, x1, t0])
mesh_explicit2 = MeshXT(x=[0., 1.], t=[0., 0.1], delta=[0.05, 0.002], boundaries=[x0, x1, t0])
mesh_implicit1 = MeshXT(x=[0., 1.], t=[0., 0.1], delta=[0.05, 0.001], boundaries=[x0, x1, t0])
mesh_implicit2 = MeshXT(x=[0., 1.], t=[0., 0.1], delta=[0.05, 0.025], boundaries=[x0, x1, t0])


# create a solver object for each case
solver_explicit1 = SolverHeatXT(mesh_explicit1, alpha=1., method='explicit')
solver_explicit2 = SolverHeatXT(mesh_explicit2, alpha=1., method='explicit')
solver_implicit1 = SolverHeatXT(mesh_implicit1, alpha=1., method='crank-nicolson')
solver_implicit2 = SolverHeatXT(mesh_implicit2, alpha=1., method='crank-nicolson')

# run the solver for each case
solver_explicit1.solver(mesh_explicit1)
solver_explicit2.solver(mesh_explicit2)
solver_implicit1.solver(mesh_implicit1)
solver_implicit2.solver(mesh_implicit2)

# calculate mean absolute error for each case
mesh_explicit1.mean_absolute_error(exact)
mesh_explicit2.mean_absolute_error(exact)
mesh_implicit1.mean_absolute_error(exact)
mesh_implicit2.mean_absolute_error(exact)

# plot u(x,t_i) at various times, t_i, for each case
mesh_explicit1.plot_solution(ntimes=3, save_to_file=False)
mesh_explicit2.plot_solution(ntimes=3, save_to_file=False)
mesh_implicit1.plot_solution(ntimes=3, save_to_file=False)
mesh_implicit2.plot_solution(ntimes=3, save_to_file=False)


