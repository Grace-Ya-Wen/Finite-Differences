import numpy as np
import matplotlib.pyplot as plt
import math


class Boundary(object):
    """
    Class that stores useful information on boundary or initial conditions for a PDE mesh.

    Attributes:
        condition (str): Type of boundary condition e.g. dirichlet, neumann, robin.
        function (function): Function used to calculate the boundary condition at a specific point.
    """
    def __init__(self, condition, function):

        # set the type of boundary condition
        self.condition = condition

        # set the boundary function/equation
        self.function = function


class MeshXY(object):
    """
    Class that stores useful information about the current XY mesh and associated boundary conditions.

    Includes methods for calculating the values along the boundaries of the mesh, as well as plotting a PDE solution
    associated with this mesh.

    Attributes:
        nx (int): Number of mesh points along the x direction.
        ny (int): Number of mesh points along the y direction.
        x (numpy array): Mesh coordinates along the x direction.
        y (numpy array): Mesh coordinates along the y direction.
        dx (float): Mesh spacing along the x direction.
        dy (float): Mesh spacing along the y direction.
        x0 (object): Instance of Boundary class representing left boundary.
        x1 (object): Instance of Boundary class representing right boundary.
        y0 (object): Instance of Boundary class representing bottom boundary.
        y1 (object): Instance of Boundary class representing top boundary.

    Arguments:
        :param x: Lower and upper limits in x direction.
        :param y: Lower and upper limits in y direction.
        :param delta: Mesh spacing in x and y directions (uniform spacing).
        :param boundaries: List of objects of the Boundary class.
        :type x: Numpy array with two elements.
        :type y: Numpy array with two elements.
        :type delta: Float.
        :type boundaries: List.
    """

    def __init__(self, x, y, delta, boundaries):

        # define the integer number of mesh points, including boundaries, based on desired mesh spacing
        self.nx = math.floor((x[1] - x[0]) / delta) + 1
        self.ny = math.floor((y[1] - y[0]) / delta) + 1

        # calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays
        self.x = np.linspace(x[0], x[1], self.nx)
        self.y = np.linspace(y[0], y[1], self.ny)

        # calculate the actual mesh spacing in x and y, should be similar or same as the dx and dy arguments
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # store the four boundary conditions as instances of the Boundary class
        self.x0 = boundaries[0]
        self.x1 = boundaries[1]
        self.y0 = boundaries[2]
        self.y1 = boundaries[3]

        # initialise solution
        self.solution = np.zeros((self.nx, self.ny))

        # apply dirichlet conditions directly to solution
        if self.x0.condition == 'dirichlet':
            self.solution[0, :] = self.x0.function(self.x[0], self.y)
        if self.x1.condition == 'dirichlet':
            self.solution[-1, :] = self.x1.function(self.x[-1], self.y)
        if self.y0.condition == 'dirichlet':
            self.solution[:, 0] = self.y0.function(self.x, self.y[0])
        if self.y1.condition == 'dirichlet':
            self.solution[:, -1] = self.y1.function(self.x, self.y[-1])

	# TODO: complete in Task 1
    def plot_pde_solution(self, save_to_file=False, save_file_name='figure.png'):
        """
        Plot the mesh PDE solution.

        Arguments:
            :param save_to_file: If True, save the plot to a file with pre-determined name.
            :param save_file_name: Name of figure to save
            :type save_to_file: Boolean.
            :type save_file_name: String.
        """

        # meshgrid for the plot
        xi, yi = np.meshgrid(self.x, self.y, indexing="ij")

        # value for each point in meshgrid
        z = self.solution
        fig, ax = plt.subplots()
        pl = ax.contourf(xi, yi, z)
        fig.colorbar(pl)
        plt.title("Temperature Distribution for 2-D Metal Sheet")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.text(2.8, 3.08, 'delta = %s'%(self.dx))

       # determine to save the figure or not
        if save_to_file:
        	plt.savefig(save_file_name, dpi = 300)
        else:
            plt.show()


class SolverEllipticXY(object):
    """
    Class containing attributes and methods useful for solving an elliptic PDE in two spatial dimensions.

    Attributes:
        nx (int) Number of mesh points to be solved (i.e. not Dirichlet boundary point) in x direction.
        ny (int): Number of mesh points to be solved (i.e. not Dirichlet boundary point) in y direction.
        n (int): Total number of mesh points to be solved across mesh.
        a (numpy array): Coefficient matrix in system of equations to solve for PDE solution.
        b (numpy array): Vector of constants in system of equations to solve for PDE solution.
        c0 (int): Lowest column number of mesh points to be solved. Associated with x direction.
        c1 (int): Higher column number of mesh points to be solved. Associated with x direction,
        r0 (int): Lowest row number of mesh points to be solved. Associated with x direction.
        r1 (int): Highest row number of mesh points to be solved. Associated with y direction.

    Arguments:
        :param poisson_function: Right-hand side equation for Poisson equation.
    """

    def __init__(self, poisson_function=None):

        # equation corresponding to forcing function
        self.poisson_function = poisson_function

        # initialise row and column start/end indices for mesh points to solve. Updated based on boundary conditions.
        self.c0 = None
        self.c1 = None
        self.r0 = None
        self.r1 = None

        # initialise attributes that may be used by the solver. Updated based on boundary conditions.
        self.nx = None
        self.ny = None
        self.n = None

        # initialise the coefficient matrix and vector of constants
        self.a = None
        self.b = None

    def solver(self, mesh, visualise=False):
        """
        Systematically run other methods used to solve an elliptic PDE for the current mesh.

        Arguments:
            :param mesh: Instance of the MeshXY class.
            :param visualise: If True, user is shown a visualisation of coefficient matrix prior to solution.
        """
        # initialise values of class attributes prior to setting A and b in system of equations
        self.initialise(mesh)

        # apply dirichlet boundaries to update the b vector
        self.dirichlet(mesh)

        # apply neumann boundaries to update the b vector
        self.neumann(mesh)

        # apply finite difference stencil to the coefficient matrix
        self.stencil()

        # apply forcing term (e.g. Poisson equation) to update the b vector if a forcing equation exists
        if self.poisson_function is not None:
            self.poisson(mesh)

        # solve the linear algebra problem for u and insert back into the mesh solution
        self.linear_algebra(mesh, visualise)

	# TODO: complete in Task 1
    def initialise(self, mesh):
        """
        Determine number of mesh points to solve in each direction and initialise the A matrix and b vector.

        Hint: mesh points associated with Dirichlet boundary conditions do not need to be solved. This can assist in
        reducing the size of the A matrix and b vector, leading to a more efficient method.

        Arguments:
            :param mesh: Instance of the MeshXY class.
        """
        y = 0 # boundary condition for the left and right boundaries. If y = 0, none of these two boundaries are dirichlet. if y = 1, one of them is and so on..
        N = (mesh.nx)*(mesh.ny)  # total number of mesh for A matrix

        # initialized the position of mesh points matrix
        # indexing. c0: the index of the first column. 
        #           c1: the index of the last column
        #           r0: the index of the first row
        #           r1: the index of the last row

        self.c0 = 0
        self.c1 = mesh.nx - 1
        self.r0 = 0
        self.r1 = mesh.ny - 1

        # initialize the mesh matrix position
        if mesh.x0.condition == 'dirichlet':
            N = N - mesh.ny
            y = y + 1 
            self.c0 += 1 

        if mesh.x1.condition == 'dirichlet':
            N = N - mesh.ny
            y = y + 1
            self.c1 -=1

        if mesh.y0.condition == 'dirichlet':
            N = N - (mesh.nx - y)
            self.r0 += 1

        if mesh.y1.condition == 'dirichlet':
            N = N - (mesh.nx - y)
            self.r1 -= 1

        # initialized the A matrix and b vector    
        self.a = np.zeros((N,N))
        self.b = np.zeros((N,1))

        self.nx = self.c1 - self.c0 + 1 # number of col for the mesh points matrix
        self.ny = self.r1 - self.r0 + 1 # number of row for the mesh points matrix

	# TODO: complete in Task 1
    def dirichlet(self, mesh):
        """
        Subtract Dirichlet boundary values from corresponding elements of the b vector.

        Arguments:
            :param mesh: Instance of the MeshXY class.
        """
        
        # update the b vector
        # run through all the mesh points and subtract Dirichlet boundary values from corresponding elements of the b vector.
        count = 0

        for i in range(self.r0, self.r1+1):
            for j in range(self.c0, self.c1+1):

                value = 0 # initilized the boundary value
                if (i == self.r0) and (mesh.y0.condition == 'dirichlet'): # bottom boundary
                    value = value + mesh.y0.function(j*mesh.dy,i*mesh.dx)
                if (j == self.c0) and (mesh.x0.condition == 'dirichlet'): # left boundary
                    value = value + mesh.x0.function(j*mesh.dy,i*mesh.dx)
                if (i == self.r1) and (mesh.y1.condition == 'dirichlet'): # top boundary
                    value = value + mesh.y1.function(j*mesh.dy,i*mesh.dx)
                if (j == self.c1) and (mesh.x1.condition == 'dirichlet'): # right boundary
                    value = value + mesh.x1.function(j*mesh.dy,i*mesh.dx)

                self.b[count,0] -= value #subtracting the dirichlet boundary value
                count = count + 1  # next element indexing
        print(self.b)
	# TODO: complete in Task 1
    def stencil(self):
        """
        Apply an appropriate finite difference stencil to each internal mesh point and update A matrix.
        """
        # number of column for mesh points matrix
        meshX = self.nx

        # update the A matrix by indexing by running through each row
        for i in range(self.b.size):
            self.a[i,i] -= 4

            if (i%meshX)+1 < meshX :      # update the matrix if a mesh point is the at the right side of the current mesh point
                self.a[i,i+1] += 1
            if (i%meshX) > 0:             # update the matrix if a mesh point is the at the left side of the current mesh point
                self.a[i,i-1] += 1
            if (i + meshX) < self.b.size: # update the matrix if there is a mesh point above the current mesh point
                self.a[i,i + meshX] += 1
            if (i - meshX) >= 0:          # update the matrix if there is a mesh point below the current mesh point
                self.a[i,i - meshX] += 1


	# TODO: complete in Task 1
    def linear_algebra(self, mesh, visualise=False):
        """
        Solve the matrix equation Au=b for the PDE solution and use to update mesh.solution.

        Arguments:
            :param mesh: Instance of the MeshXY class.
            :param visualise: If True, user is shown a visualisation of coefficient matrix prior to solution.
        """

        # optionally plot/spy the coefficient matrix - can be a useful visual check
        if visualise:
            self.plot_coefficient_matrix()

        # solve the system of equations for u
        u1d = np.linalg.solve(self.a, self.b)

        # update the solution to mesh
        count = 0
        for i in range(self.r0, self.r1+1):
            for j in range(self.c0, self.c1+1):
                mesh.solution[j,i] = u1d[count,0]
                count = count + 1
        print(u1d)
                

	# TODO: complete in Task 2
    def neumann(self, mesh):
        """
        Subtract neumann boundary value, scaled appropriately by mesh spacing, from corresponding elements of b vector.

        Arguments:
            :param mesh: Instance of the MeshXY class.
        """
        count = 0

        for i in range(self.r0, self.r1+1):
            for j in range(self.c0, self.c1+1):

                if (i == self.r0) and (mesh.y0.condition == 'neumann'): # bottom boundary
                    self.b[count,0] -= 2*mesh.dx*mesh.y0.function(j*mesh.dx,i*mesh.dy)
                    self.a[count,count+self.nx] += 1

                if (j == self.c0) and (mesh.x0.condition == 'neumann'): # left boundary
                    self.b[count,0] -= 2*mesh.dy*mesh.x0.function(j*mesh.dx,i*mesh.dy)
                    self.a[count,count+1] += 1

                if (i == self.r1) and (mesh.y1.condition == 'neumann'): # top boundary
                    self.b[count,0] -= 2*mesh.dy*mesh.y1.function(j*mesh.dx,i*mesh.dx)
                    self.a[count,count-self.nx] += 1

                if (j == self.c1) and (mesh.x1.condition == 'neumann'): # right boundary
                    self.b[count,0] -= 2*mesh.dx*mesh.x1.function(j*mesh.dx,i*mesh.dy)
                    self.a[count,count-1] += 1
                count = count + 1  # next element indexing


	# TODO: complete in Task 2
    def poisson(self, mesh):
        """
        Calculate the forcing term, f(x,y), in Poisson equation and update corresponding elements of b vector.

        Arguments:
            :param mesh: Instance of the MeshXY class.
        """
        # update the b vector
        count = 0
        for i in range(self.r0, self.r1+1):
            for j in range(self.c0, self.c1+1):
                self.b[count,0] += ((mesh.dx)**2) * self.poisson_function(j*mesh.dx,i*mesh.dy)
                count = count + 1


    def plot_coefficient_matrix(self):
        """
        Visualise the non-zero entries in the coefficient matrix A.
        """
        fig, ax = plt.subplots(1, 1)
        ax.spy(self.a, markersize=5)
        plt.show()
