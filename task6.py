# import statements
from functions_part2 import *


# define the boundary and initial condition functions
def left(x, t): return 0.*x+0.*t
def right(x, t): return 0.*x+0.*t
def initial(x, t): return np.sin(np.pi*x)+0.*t
def initial_derivative(x, t): return 0.*x + 0.*t


# define the analytic solution function
def exact(x, t): return np.sin(np.pi*x)*np.cos(np.pi*t)


# define the boundaries properties and create each BoundariesXY object
x0 = Boundary('dirichlet', left)
x1 = Boundary('dirichlet', right)
t0 = Boundary('initial', initial)
dt0 = Boundary('initial_derivative', initial_derivative)

# create a mesh object for each solution approach
mesh = MeshXT(x=[0., 1.], t=[0., 1.], delta=[0.01, 0.01], boundaries=[x0, x1, t0, dt0])

# run the wave equation solver function
solver_wave_xt(mesh)

# plot the wave equation solution
mesh.plot_solution(ntimes=6, save_to_file=False)


