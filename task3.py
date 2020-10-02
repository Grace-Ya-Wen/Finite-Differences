# import statements
from functions_part1 import *

# define the boundary functions
def left(x, y): return 0. * x + 0. * y
def right(x, y): return 0. * x + 6. * y * y
def bottom(x, y): return 0. * x + 0. * y
def top(x, y): return 9. * x * x + 0. * y

# define the poisson function
def poisson_function(x, y): return 2. * (x * x + y * y)

# define the boundaries properties and create each BoundariesXY object
x0 = Boundary('dirichlet', left)
x1 = Boundary('neumann', right)
y0 = Boundary('dirichlet', bottom)
y1 = Boundary('dirichlet', top)

# create an instance of the MeshXY class
mesh = MeshXY(x=[0., 3.], y=[0., 3.], delta=1.
, boundaries=[x0, x1, y0, y1])
# create an instance of the EllipticXY class
elliptic = SolverEllipticXY(poisson_function)

# evaluate the solution using the solver
elliptic.solver(mesh, visualise=True)

# plot the solution to the PDE on the mesh
mesh.plot_pde_solution(save_to_file=False)