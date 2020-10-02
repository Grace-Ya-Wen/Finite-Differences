# import statements
from functions_part1 import *


# define the boundary functions
def left(x, y): return 0. * x + 0. * y
def right(x, y): return 0. * x + 3. * y * y * y
def bottom(x, y): return 0. * x + 0. * y
def top(x, y): return 6. * x + 0. * y


# define the poisson function
def poisson_function(x, y): return 12. * x * y


# define the boundaries properties and create each BoundariesXY object
x0 = Boundary('dirichlet', left)
x1 = Boundary('dirichlet', right)
y0 = Boundary('dirichlet', bottom)
y1 = Boundary('neumann', top)

# create an instance of the MeshXY class
mesh = MeshXY(x=[0., 1.5], y=[0., 1.], delta=0.1, boundaries=[x0, x1, y0, y1])
# create an instance of the EllipticXY class
elliptic = SolverEllipticXY(poisson_function)

# evaluate the solution using the solver
elliptic.solver(mesh, visualise=True)

# plot the solution to the PDE on the mesh
mesh.plot_pde_solution(save_to_file=False)
