# import statements
from functions_part1 import *


# define the boundary functions
def left(x, y): return 100. + 0. * x + 0. * y
def right(x, y): return 100. + 0. * x + 0. * y
def bottom(x, y): return 100. + 0. * x + 0. * y
def top(x, y): return 0. + 0. * x + 0. * y


# define the boundaries properties and create each BoundariesXY object
x0 = Boundary('dirichlet', left)
x1 = Boundary('dirichlet', right)
y0 = Boundary('dirichlet', bottom)
y1 = Boundary('dirichlet', top)

# create an instance of the MeshXY class
mesh = MeshXY(x=[0., 12.], y=[0., 12.], delta=1, boundaries=[x0, x1, y0, y1])

# create an instance of the EllipticXY class
elliptic = SolverEllipticXY()

# evaluate the solution using the solver
elliptic.solver(mesh, visualise=True)

# plot the PDE solution
mesh.plot_pde_solution(save_to_file=False)
