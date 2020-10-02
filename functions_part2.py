import numpy as np
import matplotlib.pyplot as plt
import math


class Boundary(object):
    """
    Class that stores useful information on boundary or initial conditions for a PDE mesh.

    Attributes:
        condition (str): Type of boundary condition e.g. dirichlet, neumann, robin.
        function (function): Function used to calculate the boundary value at a specific point.
    """
    def __init__(self, condition, function):

        # set the type of boundary condition
        self.condition = condition

        # set the boundary function/equation
        self.function = function


class MeshXT(object):
    """
    Class that stores useful information about the current XT mesh and associated boundary conditions.

    Includes methods for plotting the PDE solution associated with this mesh and calculating the error relative
    to the analytic solution, if it is known.

    Attributes:
        nx (int): Number of mesh points along the x dimension.
        nt (int): Number of mesh points along the t dimension.
        x (numpy array): Mesh coordinates along the x dimension.
        t (numpy array): Mesh coordinates along the t dimension.
        dx (float): Mesh spacing along the x dimension.
        dt (float): Mesh spacing along the t dimension.
        boundaries (list): List of objects of Boundary class. Order is u(x0,t), u(x1,t), u(x,t0), du(x,t0)/dt, etc...
        solution (numpy array): PDE solution as calculated on the mesh.
        method (string): Name of the solution method e.g. explicit, crank-nicolson, galerkin, backward-euler.
        mae (float): mean absolute error of numerical/mesh solution relative to analytic/exact solution.

    Arguments:
        :param x: Lower and upper limits in x dimension.
        :param t: Lower and upper limits in t dimension.
        :param delta: Mesh spacing in x and t dimensions.
        :param boundaries: List of objects of the Boundary class, with same order as class attribute of same name.
        :type x: Numpy array with two elements.
        :type t: Numpy array with two elements.
        :type delta: Numpy array with two elements.
        :type boundaries: List.
    """

    def __init__(self, x, t, delta, boundaries):

        # define the integer number of mesh points, including boundaries, based on desired mesh spacing
        self.nx = math.floor((x[1] - x[0]) / delta[0]) + 1
        self.nt = math.floor((t[1] - t[0]) / delta[1]) + 1

        # calculate the x and y values/coordinates of mesh as one-dimensional numpy arrays
        self.x = np.linspace(x[0], x[1], self.nx)
        self.t = np.linspace(t[0], t[1], self.nt)

        # calculate the actual mesh spacing in x and y, should be similar or same as the dx and dy arguments
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

        # store the boundary and initial condition as a list of Boundary class objects
        self.boundaries = boundaries

        # initialise method name - useful for plot title. Updated by solver.
        self.method = None

        # initialise mean absolute error to be None type.
        self.mae = None

        # initialise the full PDE solution matrix
        self.solution = np.zeros((self.nx, self.nt))

        # apply dirichlet boundary conditions directly to the mesh solution
        if self.boundaries[0].condition == 'dirichlet':
            self.solution[0, :] = self.boundaries[0].function(self.x[0], self.t)
        if self.boundaries[1].condition == 'dirichlet':
            self.solution[-1, :] = self.boundaries[1].function(self.x[-1], self.t)

        # apply initial conditions directly to the mesh solution
        self.solution[:, 0] = self.boundaries[2].function(self.x, self.t[0])

    # TODO: complete in Task 4
    def plot_solution(self, ntimes, save_to_file=False, save_file_name='figure.png'):
        """
        Plot the mesh solution u(x,t^n) at a fixed number of time steps.

        Arguments:
            :param ntimes: number of times at which to plot the solution, u(x,t_n).
            :param save_to_file: If True, save the plot to a file with pre-determined name.
            :param save_file_name: Name of figure to save.
            :type ntimes: Integer.
            :type save_to_file: Boolean.
            :type save_file_name: String.
        """
        step = math.floor((self.nt)/(ntimes-1))
        for i in range(0,self.nt,step):
            plt.plot(self.x,self.solution[:,i], label =r"time={}s".format(i*self.dt))
            #plt.text(0.0, 1.0, 'error = %s'%(self.mae))
            #plt.text(0.0, 0.9, 'delta_t = %s'%(self.dt))
            plt.xlabel("x")
            plt.ylabel("Temperature")
            #plt.ylabel("Vertical Displacement")

        plt.title('Temperature Along x-axis Over Time')
        #plt.title('Vertical Displacement Along x-axis Over Time')

        plt.legend()


        # determine to save the figure or not
        if save_to_file:
        	plt.savefig(save_file_name, dpi = 300)
        else:
            plt.show()

    # TODO: complete in Task 5
    def mean_absolute_error(self, exact):
        """
        Calculates the mean absolute error in the solution, relative to exact solution, for the final time step.

        Arguments:
            :param exact: The exact solution to the PDE.
            :type exact: Function.
        """
        error = 0 # initliad total error
        count = 0 # number of evaluated points

        for i in range(self.nx):
            for j in range(self.nt):
                error += abs(self.solution[i,j] - exact(self.dx*i,self.dt*j))
                count = count + 1
        self.mae = error/count
        


class SolverHeatXT(object):
    """
    Class containing attributes and methods useful for solving the 1D heat equation.

    Attributes:
        method (string): Name of the solution method e.g. crank-nicolson, explicit, galerkin, backward-euler.
        r (float): ratio of mesh spacings, useful for evaluating stability of solution method.
        theta (float): Weighting factor used in implicit scheme e.g. Crank-Nicolson has theta=1/2.
        a (numpy array): Coefficient matrix in system of equations to solve for PDE solution.
        b (numpy array): Vector of constants in system of equations to solve for PDE solution at time t^n.

    Arguments:
        :param mesh: instance of the MeshXT class.
        :param alpha: reciprocal of thermal diffusivity parameter in the heat equation.
        :param method: Name of the solution method e.g. crank-nicolson, explicit, galerkin, backward-euler.
        :type mesh: Object.
        :type alpha: Float.
        :type method: String.
    """

    def __init__(self, mesh, alpha=1., method='crank-nicolson'):

        # set the solution method
        self.method = method
        mesh.method = method

        # set the ratio of mesh spacings used in solver equations
        self.r = mesh.dt / (alpha * mesh.dx * mesh.dx)

        # initialise theta variable used in the implicit methods
        self.theta = None

        # determine if solution method requires matrix equation and set A, b accordingly
        if self.method == 'explicit':
            self.a = None
            self.b = None
        else:
            self.a = np.zeros((mesh.nx*2, mesh.nx*2))
            self.b = np.zeros(mesh.nx*2)

    def solver(self, mesh):
        """
        Run the requested solution method. Default to Crank-Nicolson implicit method if user hasn't specified.

        Arguments:
            :param mesh: Instance of the MeshXT class.
        """

        # run the explicit solution method
        if self.method == 'explicit':
            self.explicit(mesh)

        # run the crank-nicolson implicit method
        elif self.method == 'crank-nicolson':
            self.theta = 0.5
            self.implicit(mesh)

        elif self.method == 'galerkin':
            self.theta = 2./3.
            self.implicit(mesh)

        elif self.method == 'backward-euler':
            self.theta = 1.
            self.implicit(mesh)

        else:  # default to crank-nicolson
            self.method = 'crank-nicolson'
            mesh.method = 'crank-nicolson'
            self.theta = 0.5
            self.implicit(mesh)

    # TODO: complete in Task 4
    def explicit(self, mesh):
        """
        Solve the 1D heat equation using an explicit scheme.

        Arguments:
            :param mesh: Instance of the MeshXT class.
        """ 
        # run through all the unknown mesh points
        for i in range(1,mesh.nt):
            for j in range(1,mesh.nx-1):
                mesh.solution[j,i] = self.r*mesh.solution[j-1,i-1] + (1-2*self.r)*mesh.solution[j,i-1]+self.r*mesh.solution[j+1,i-1]

    # TODO: complete in Task 4
    def implicit(self, mesh):
        """
        Solve the 1D heat equation using an implicit scheme. Accounts for different values of theta.

        Arguments:
            :param mesh: Instance of the MeshXT class.
        """
        a = np.zeros((mesh.nx-2,mesh.nx-2))
        # run through all the rows
        for j in range(mesh.nx-2):
            a[j,j] = -(1+self.r)
            if j > 0:
                a[j,j-1] = self.theta*self.r
            if j < mesh.nx-3 and mesh.nx>3:
                a[j,j+1] = self.theta*self.r

        for i in range (1,mesh.nt):
            b = np.zeros((mesh.nx-2,1))     # initialized b vector
            count = 0                       # indexing 
            # run through each row of a specified 
            for j in range(1,mesh.nx-1):
                b[count,0] = -(1-self.theta)*self.r * mesh.solution[j-1,i-1] - (1-2*self.r*self.theta)*mesh.solution[j,i-1]-(1-self.theta)*self.r *mesh.solution[j+1,i-1]
                if j == 1:                  # for the first column
                    b[count,0] -= self.theta*self.r*mesh.solution[0,i]         # substracing additional value from vector b
                elif j == mesh.nx - 2:      # for the last column
                    b[count,0] -= self.theta*self.r*mesh.solution[mesh.nx-1,i]  # substracing additional value from vector b

                count = count + 1
            u1d = np.linalg.solve(a,b)
            
            # change value in solution matrix
            for n in range (1,mesh.nx-1):
                mesh.solution[n,i] = u1d[n-1]
            
            

# TODO: complete in Task 6
def solver_wave_xt(mesh):
    """
    Function used to solve the 1D wave equation (i.e. hyperbolic PDE). Assumes dx = dt and an explicit method.

    Has no return. Instead, it updates the mesh.solution attribute directly.

    Arguments:
        :param mesh: instance of the MeshXT class.
    """
    #  run through all the unknown mesh points
    for i in range(1,mesh.nt):
        for j in range(1,mesh.nx-1):
            if i == 1:    # solution for initial boundary 
                mesh.solution[j,i] = 0.5*(mesh.solution[j-1,i-1] +mesh.solution[j+1,i-1]) + mesh.dt*mesh.boundaries[3].function(j*mesh.dx,(i-1)*mesh.dt)
            else:         # solution for others
                mesh.solution[j,i] = mesh.solution[j-1,i-1]+mesh.solution[j+1,i-1]- mesh.solution[j,i-2]
    

