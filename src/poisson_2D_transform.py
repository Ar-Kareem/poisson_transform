import logging
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

class Transformation:
    def __init__(self, ksi, eta, Tx, Ty):
        self.ksi = ksi
        self.eta = eta
        self.f_Tx = Tx
        self.f_Ty = Ty
        self.f_dx_dksi = self.f_Tx.diff(ksi)
        self.f_dx_deta = self.f_Tx.diff(eta)
        self.f_dy_dksi = self.f_Ty.diff(ksi)
        self.f_dy_deta = self.f_Ty.diff(eta)
        self.f_dx_dksiksi = self.f_dx_dksi.diff(ksi)
        self.f_dx_dksieta = self.f_dx_dksi.diff(eta)
        self.f_dx_detaeta = self.f_dx_deta.diff(eta)
        self.f_dy_dksiksi = self.f_dy_dksi.diff(ksi)
        self.f_dy_dksieta = self.f_dy_dksi.diff(eta)
        self.f_dy_detaeta = self.f_dy_deta.diff(eta)
    # transformations
    def Tx(self, ksi_val, eta_val):
        return float(self.f_Tx.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def Ty(self, ksi_val, eta_val):
        return float(self.f_Ty.subs({self.ksi: ksi_val, self.eta: eta_val}))
    # first derivatives
    def dx_dksi(self, ksi_val, eta_val):
        return float(self.f_dx_dksi.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dx_deta(self, ksi_val, eta_val):
        return float(self.f_dx_deta.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dy_dksi(self, ksi_val, eta_val):
        return float(self.f_dy_dksi.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dy_deta(self, ksi_val, eta_val):
        return float(self.f_dy_deta.subs({self.ksi: ksi_val, self.eta: eta_val}))
    # second derivatives
    def dx_dksiksi(self, ksi_val, eta_val):
        return float(self.f_dx_dksiksi.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dx_dksieta(self, ksi_val, eta_val):
        return float(self.f_dx_dksieta.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dx_detaeta(self, ksi_val, eta_val):
        return float(self.f_dx_detaeta.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dy_dksiksi(self, ksi_val, eta_val):
        return float(self.f_dy_dksiksi.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dy_dksieta(self, ksi_val, eta_val):
        return float(self.f_dy_dksieta.subs({self.ksi: ksi_val, self.eta: eta_val}))
    def dy_detaeta(self, ksi_val, eta_val):
        return float(self.f_dy_detaeta.subs({self.ksi: ksi_val, self.eta: eta_val}))

    def take_first_order_derivatives(self, ksi_val, eta_val):
        dx_dksi = self.dx_dksi(ksi_val, eta_val)
        dx_deta = self.dx_deta(ksi_val, eta_val)
        dy_dksi = self.dy_dksi(ksi_val, eta_val)
        dy_deta = self.dy_deta(ksi_val, eta_val)
        return dx_dksi, dx_deta, dy_dksi, dy_deta
    def take_second_order_derivatives(self, ksi_val, eta_val):
        dx_dksiksi = self.dx_dksiksi(ksi_val, eta_val)
        dx_dksieta = self.dx_dksieta(ksi_val, eta_val)
        dx_detaeta = self.dx_detaeta(ksi_val, eta_val)
        dy_dksiksi = self.dy_dksiksi(ksi_val, eta_val)
        dy_dksieta = self.dy_dksieta(ksi_val, eta_val)
        dy_detaeta = self.dy_detaeta(ksi_val, eta_val)
        return dx_dksiksi, dx_dksieta, dx_detaeta, dy_dksiksi, dy_dksieta, dy_detaeta


    @staticmethod
    def get_ksi_eta():
        import sympy
        ksi, eta = sympy.symbols('ksi eta', real=True)
        return ksi, eta

def get_identity_transformation():
    ksi, eta = Transformation.get_ksi_eta()
    Tx = ksi
    Ty = eta
    return Transformation(ksi, eta, Tx, Ty)

def _stamp_derivative(matrix, nodemap, i, j, dimension, value):
    """ Stamp the matrix to take derivatives, coefs are from https://en.wikipedia.org/wiki/Finite_difference_coefficient """
    Nx, Ny = nodemap.shape
    x_sign = {0: 1, (Nx-1): -1}.get(i, 0)  # 0 if not on left/right boundary, 1 if on left, -1 if on right, if 0 then easy ksi derivative otherwise harder due to ghost point
    y_sign = {0: 1, (Ny-1): -1}.get(j, 0)  # 0 if not on top/bottom boundary, 1 if on top, -1 if on bottom, if 0 then easy eta derivative otherwise harder due to ghost point
    if dimension == 'x':
        if x_sign == 0:
            matrix[nodemap[i, j], nodemap[i+1, j]] += +value/2  # TODO is this 1/2 correct?
            matrix[nodemap[i, j], nodemap[i-1, j]] += -value/2
        else:
            # FORWARD: −3f(x) + 4f(x + ∆x) − f(x + 2∆x) /(2∆x)
            # BACKWARD: 3f(x) − 4f(x − ∆x) + f(x − 2∆x) /(2∆x)
            matrix[nodemap[i, j], nodemap[i, j]]          += -3*x_sign*value/2
            matrix[nodemap[i, j], nodemap[i+x_sign, j]]   +=  4*x_sign*value/2
            matrix[nodemap[i, j], nodemap[i+2*x_sign, j]] += -1*x_sign*value/2
    elif dimension == 'y':
        if y_sign == 0:
            matrix[nodemap[i, j], nodemap[i, j+1]] += +value/2
            matrix[nodemap[i, j], nodemap[i, j-1]] += -value/2
        else:
            # FORWARD: −3f(x) + 4f(x + ∆x) − f(x + 2∆x) /(2∆x)
            # BACKWARD: 3f(x) − 4f(x − ∆x) + f(x − 2∆x) /(2∆x)
            matrix[nodemap[i, j], nodemap[i, j]]          += -3*y_sign*value/2
            matrix[nodemap[i, j], nodemap[i, j+y_sign]]   +=  4*y_sign*value/2
            matrix[nodemap[i, j], nodemap[i, j+2*y_sign]] += -1*y_sign*value/2
    elif dimension == 'xx':
        if x_sign == 0:
            matrix[nodemap[i, j], nodemap[i+1, j  ]] += +value
            matrix[nodemap[i, j], nodemap[i  , j  ]] += -2*value
            matrix[nodemap[i, j], nodemap[i-1, j  ]] += +value
        else:
            # FORWARD: 2f(x) − 5f(x + ∆x) + 4f(x + 2∆x) − f(x + 3∆x) /∆x^2
            # BACKWARD: -2f(x) + 5f(x − ∆x) − 4f(x − 2∆x) + f(x − 3∆x) /∆x^2
            matrix[nodemap[i, j], nodemap[i, j]]          += +2*x_sign*value
            matrix[nodemap[i, j], nodemap[i+x_sign, j]]   += -5*x_sign*value
            matrix[nodemap[i, j], nodemap[i+2*x_sign, j]] += +4*x_sign*value
            matrix[nodemap[i, j], nodemap[i+3*x_sign, j]] += -1*x_sign*value
    elif dimension == 'yy':
        if y_sign == 0:
            matrix[nodemap[i, j], nodemap[i  , j+1]] += +value
            matrix[nodemap[i, j], nodemap[i  , j  ]] += -2*value
            matrix[nodemap[i, j], nodemap[i  , j-1]] += +value
        else:
            # FORWARD: 2f(x) − 5f(x + ∆x) + 4f(x + 2∆x) − f(x + 3∆x) /∆x^2
            # BACKWARD: -2f(x) + 5f(x − ∆x) − 4f(x − 2∆x) + f(x − 3∆x) /∆x^2
            matrix[nodemap[i, j], nodemap[i, j]]          += +2*y_sign*value
            matrix[nodemap[i, j], nodemap[i, j+y_sign]]   += -5*y_sign*value
            matrix[nodemap[i, j], nodemap[i, j+2*y_sign]] += +4*y_sign*value
            matrix[nodemap[i, j], nodemap[i, j+3*y_sign]] += -1*y_sign*value
    elif dimension == 'xy':
        if x_sign == 0 and y_sign == 0:
            matrix[nodemap[i, j], nodemap[i+1, j+1]] += +value/4
            matrix[nodemap[i, j], nodemap[i+1, j-1]] += -value/4
            matrix[nodemap[i, j], nodemap[i-1, j+1]] += -value/4
            matrix[nodemap[i, j], nodemap[i-1, j-1]] += +value/4
        else:
            raise NotImplementedError("Not implemented")
    else:
        raise Exception("Should not be here")

def SolvePoisson(Nx: int, Ny: int, transformation: Transformation = None, f: callable = None, g: callable = None):
    """Returns matrix A and vector b such that solving for u in Au=b is u that solves the Poisson equation
    f is the right hand side in the poisson equation such that:
        -∇^2 u = f in Ω   where Ω = Tx([0, 1], [0, 1])xTy([0, 1], [0, 1])  and   f(x, y) -> f
    g is the boundary condition such that:
        a*u + b*∂u/∂n = g on ∂Ω   where g(x, y) -> (a, b, g)  when x, y are on the boundary of Ω
    g should return None otherwise
    With possibly Dirichlet boundary conditions inside Ω   where g(x, y) -> (1, 0, u)  when x, y are inside Ω
    """
    if transformation is None:
        transformation = get_identity_transformation()
    if f is None and g is None:
        logger.warning("Warning: both f and g are None. Are you sure this is what you want?")
    if f is None:
        logger.info("f is None. Setting f=1")
        f = lambda x, y, ksi, eta: 1
    if g is None:
        logger.info("g is None. Setting Dirichlet BCs ")
        g = lambda x, y, ksi, eta: (1, 0, 0) if (ksi in (0, 1) or eta in (0, 1)) else None

    d_ksi, d_eta = 1/(Nx-1), 1/(Ny-1)
    ksi, eta = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
    NodeMap = np.arange(0, Nx*Ny).reshape((Nx, Ny), order='F')
    A = scipy.sparse.lil_matrix((Nx*Ny, Nx*Ny))
    b_rhs = np.zeros(Nx*Ny)
    J = np.zeros((Nx, Ny))
    # Matrices to take first order derivatives of u
    Dx_1st_order = scipy.sparse.lil_matrix((Nx*Ny, Nx*Ny))
    Dy_1st_order = scipy.sparse.lil_matrix((Nx*Ny, Nx*Ny))
    for i in range(Nx):
        for j in range(Ny): 
            ksi[i, j] = i*d_ksi
            eta[i, j] = j*d_eta
            if i == Nx-1:
                ksi[i, j] = 1  # prevent numerical errors
            if j == Ny-1:
                eta[i, j] = 1
            J[i, j] = transformation.dx_dksi(ksi[i, j], eta[i, j]) * transformation.dy_deta(ksi[i, j], eta[i, j]) \
                        - transformation.dx_deta(ksi[i, j], eta[i, j]) * transformation.dy_dksi(ksi[i, j], eta[i, j])
            assert np.abs(J[i, j]) > 0, "Jacobian is zero (J={}) at i={}, j={}, dx_dksi={}, dy_deta={}, dx_deta={}, dy_dksi={}".format(
                                    J[i, j], i, j, transformation.dx_dksi(ksi[i, j], eta[i, j]), transformation.dy_deta(ksi[i, j], eta[i, j]), 
                                    transformation.dx_deta(ksi[i, j], eta[i, j]), transformation.dy_dksi(ksi[i, j], eta[i, j]))

    for i in range(Nx):
        for j in range(Ny):
            b_rhs[NodeMap[i, j]] = f(ksi=ksi[i, j], eta=eta[i, j], x=transformation.Tx(ksi[i, j], eta[i, j]), y=transformation.Ty(ksi[i, j], eta[i, j]))
            dx_dksi, dx_deta, dy_dksi, dy_deta = transformation.take_first_order_derivatives(ksi[i, j], eta[i, j])
            dx_dksiksi, dx_dksieta, dx_detaeta, dy_dksiksi, dy_dksieta, dy_detaeta = transformation.take_second_order_derivatives(ksi[i, j], eta[i, j])

            a = (dx_deta**2 + dy_deta**2)
            b = (dx_dksi*dx_deta + dy_dksi*dy_deta)
            c = (dx_dksi**2 + dy_dksi**2)
            alpha = a*dx_dksiksi - 2*b*dx_dksieta + c*dx_detaeta
            beta = a*dy_dksiksi - 2*b*dy_dksieta + c*dy_detaeta
            d = 1/J[i, j] * (beta*dx_deta - alpha*dy_deta)
            e = 1/J[i, j] * (alpha*dy_dksi - beta*dx_dksi)

            inv_J_sq = -1/J[i, j]**2
            c_ksiksi = inv_J_sq * a
            c_ksieta = inv_J_sq * -2 * b
            c_etaeta = inv_J_sq * c
            c_ksi = inv_J_sq * d
            c_eta = inv_J_sq * e

            # setup first order derivative matrix, not needed for solving poissons equation but useful for plotting
            # ux = 1/J (yηuξ − yξuη)
            cx_ksi = 1/J[i, j] * dy_deta
            cx_eta = 1/J[i, j] * -dy_dksi
            # uy = 1/J (−xηuξ + xξuη)
            cy_ksi = 1/J[i, j] * -dx_deta
            cy_eta = 1/J[i, j] * dx_dksi
            # take derivatives, making sure to not go out of bounds and maintain second order convergence
            # dudx matrix
            _stamp_derivative(Dx_1st_order, NodeMap, i, j, 'x', value=cx_ksi / d_ksi)
            _stamp_derivative(Dx_1st_order, NodeMap, i, j, 'y', value=cx_eta / d_eta)
            # dudy matrix
            _stamp_derivative(Dy_1st_order, NodeMap, i, j, 'x', value=cy_ksi / d_ksi)
            _stamp_derivative(Dy_1st_order, NodeMap, i, j, 'y', value=cy_eta / d_eta)

            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:  # skip boundary nodes
                continue

            # INTERIOR POINTS
            _stamp_derivative(A, NodeMap, i, j, 'xx', value=c_ksiksi / (d_ksi**2))  # u_ksiksi
            _stamp_derivative(A, NodeMap, i, j, 'yy', value=c_etaeta / (d_eta**2))  # u_etaeta
            _stamp_derivative(A, NodeMap, i, j, 'xy', value=c_ksieta / (d_ksi*d_eta))  # u_ksieta
            _stamp_derivative(A, NodeMap, i, j, 'x', value=c_ksi / d_ksi)  # u_ksi
            _stamp_derivative(A, NodeMap, i, j, 'y', value=c_eta / d_eta)  # u_eta

    laplacian_matrix = A.copy()  # TODO not really laplacian matrix as the boundaries are not stamped

    for i in range(Nx):
        for j in range(Ny):
            if i != 0 and i != Nx-1 and j != 0 and j != Ny-1:  # skip interior nodes
                continue
            # α*u + β*∂u/∂n = g on ∂Ω
            alpha, beta, g_val = g(ksi=ksi[i, j], eta=eta[i, j], x=transformation.Tx(ksi[i, j], eta[i, j]), y=transformation.Ty(ksi[i, j], eta[i, j]))
            # g
            b_rhs[NodeMap[i, j]] = g_val
            # α*u
            A[NodeMap[i, j], NodeMap[i, j]] += alpha
            # β*∂u/∂n  (we need to compute ∂u/∂n)
            dx_dksi, dx_deta, dy_dksi, dy_deta = transformation.take_first_order_derivatives(ksi[i, j], eta[i, j])
            # parallel to current edge
            if j == 0:
                parallel = (-dx_dksi, -dy_dksi)
            elif j == Ny-1:
                parallel = (dx_dksi, dy_dksi)
            elif i == 0:
                parallel = (dx_deta, dy_deta)
            elif i == Nx-1:
                parallel = (-dx_deta, -dy_deta)
            else:
                raise Exception("Should not be here")
            n__norm = np.sqrt(parallel[0]**2 + parallel[1]**2)
            n__x = -parallel[1]/n__norm
            n__y = parallel[0]/n__norm
            # β * ∂u/∂n = β* u_x n__x + β* u_y n__y  =  β/J [(y_η n__x − x_η n__y) u_ξ + (-y_ξ n__x + x_ξ n__y) u_η]
            # c_ksi = β/J (y_η n__x - x_η n__y )
            # c_η = β/J (x_ξ n__y - y_ξ n__x )
            c_ksi = beta/J[i, j] * (dy_deta*n__x - dx_deta*n__y)
            c_eta = beta/J[i, j] * (dx_dksi*n__y - dy_dksi*n__x)

            _stamp_derivative(A, NodeMap, i, j, 'x', c_ksi/d_ksi)
            _stamp_derivative(A, NodeMap, i, j, 'y', c_eta/d_eta)

    # impose Dirichlet conditions inside Ω (essentially when we know the value of u at certain points)
    F = b_rhs.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            g_ret = g(ksi=ksi[i, j], eta=eta[i, j], x=transformation.Tx(ksi[i, j], eta[i, j]), y=transformation.Ty(ksi[i, j], eta[i, j]))
            if g_ret is None:
                continue
            alpha, beta, g_val = g_ret
            assert alpha == 1 and beta == 0, "Only Dirichlet BCs are supported inside Ω"
            A[NodeMap[i, j], :] = 0
            A[NodeMap[i, j], NodeMap[i, j]] = 1
            b_rhs[NodeMap[i, j]] = g_val
    return {'A': A, 'b_rhs': b_rhs, 'laplacian_matrix': laplacian_matrix, 'F': F, 
            'Dx_1st_order': Dx_1st_order, 'Dy_1st_order': Dy_1st_order, 'Jacobian': J} 

def solveLinear(Nx, Ny, A, b_rhs):
    solution = scipy.sparse.linalg.spsolve(A.tocsr(), b_rhs)
    return solution.reshape((Nx, Ny), order='F')

def integrateSolution(Nx, Ny, solution, jacobian):
    d_ksi, d_eta = 1/(Nx-1), 1/(Ny-1)
    sol_integral = 0
    for j in range(Ny-1):
        for i in range(Nx-1):
            sol_integral += np.mean(solution[i:i+2, j:j+2]) * np.mean(jacobian[i:i+2, j:j+2])*d_ksi*d_eta
    return sol_integral

def plotGeometry(Nx: int, Ny: int, transformation: Transformation = None):
    if transformation is None:
        transformation = get_identity_transformation()
    ksi, eta = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), indexing='ij')
    x = np.array([transformation.Tx(ksi_val, eta_val) for ksi_val, eta_val in zip(ksi.flatten(), eta.flatten())]).reshape(ksi.shape)
    y = np.array([transformation.Ty(ksi_val, eta_val) for ksi_val, eta_val in zip(ksi.flatten(), eta.flatten())]).reshape(ksi.shape)
    plt.figure(figsize=(6, 5))
    # subplot size
    plt.plot(x, y, 'k')
    plt.plot(x.T, y.T, 'k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.axis('equal')
    plt.title('Geometry')


def plotSolution(Nx: int, Ny: int, solution: np.ndarray, transformation: Transformation = None, contour_levels=20, gradient: (np.ndarray, np.ndarray) = None, laplace_u: np.ndarray = None):
    if transformation is None:
        transformation = get_identity_transformation()
    ksi, eta = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), indexing='ij')
    x = np.array([transformation.Tx(ksi_val, eta_val) for ksi_val, eta_val in zip(ksi.flatten(), eta.flatten())]).reshape(ksi.shape)
    y = np.array([transformation.Ty(ksi_val, eta_val) for ksi_val, eta_val in zip(ksi.flatten(), eta.flatten())]).reshape(ksi.shape)
    num_plots = 1 + (gradient is not None) + (laplace_u is not None)
    plot_idx = 1
    plt.figure(figsize=(6*num_plots, 5))
    if num_plots > 1:
        plt.subplot(1, num_plots, plot_idx)
    # segs1 = np.stack((x, y), axis=2)
    # segs2 = segs1.transpose(1, 0, 2)
    # plt.gca().add_collection(LineCollection(segs1, alpha=0.2, colors='white'))
    # plt.gca().add_collection(LineCollection(segs2, alpha=0.2, colors='white'))
    plt.contourf(x, y, solution, 41, cmap='inferno')
    plt.colorbar()
    plt.contour(x, y, solution, contour_levels, colors='k', linewidths=0.2)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution $u$')

    if gradient is not None:
        plot_idx += 1
        plt.subplot(1, num_plots, plot_idx)
        plotStreams(x, y, gradient[1], gradient[2], gradient[0])
    
    if laplace_u is not None:
        plot_idx += 1
        plt.subplot(1, num_plots, plot_idx)
        plt.contourf(x, y, laplace_u, 41, cmap='inferno')
        plt.colorbar()
        plt.contour(x, y, laplace_u, contour_levels, colors='k', linewidths=0.2)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(r'$-\nabla^2 u$')
    plt.tight_layout()


def plotStreams(xx, yy, u, v, z, ax=None):
    if ax is None:
        ax = plt.gca()

    from scipy.interpolate import griddata
    x = np.linspace(xx.min(), xx.max(), 50)
    y = np.linspace(yy.min(), yy.max(), 50)
    xi, yi = np.meshgrid(x,y)

    # interpolate data onto grid:
    px = xx.flatten(order='F')
    py = yy.flatten(order='F')
    pu = u.flatten(order='F')
    pv = v.flatten(order='F')
    pz = z.flatten(order='F')

    zipped = np.r_[ px[None,:], py[None,:] ].T
    gu = griddata(zipped, pu, (xi,yi))
    gv = griddata(zipped, pv, (xi,yi))
    gz = griddata(zipped, pz, (xi,yi))
    lw = 2*gz/np.nanmax(gz)

    im = ax.contourf(xx, yy, z, 41, cmap='afmhot')
    plt.colorbar(im, ax=ax)
    # geometry
    # ax.plot(xx, yy,'-k', alpha=0.3)
    # ax.plot(xx.T, yy.T,'-k', alpha=0.3)
    # grid
    # ax.plot(xi, yi, '-b', alpha=0.1)
    # ax.plot(xi.T, yi.T, '-b', alpha=0.1)
    ax.streamplot(x, y, gu, gv, color='w', density=2, linewidth=lw)
    ax.axis('equal')
    plt.title(r'$|-\nabla u|$')

def solve_and_plot(Nx, Ny, transformation=None, f=None, g=None, contour_levels=20):
    d = SolvePoisson(Nx, Ny, transformation=transformation, f=f, g=g)
    sol = solveLinear(Nx, Ny, d['A'], d['b_rhs'])
    _sol_flat = sol.flatten(order='F')
    du_dx = d['Dx_1st_order'].dot(_sol_flat)
    du_dy = d['Dy_1st_order'].dot(_sol_flat)
    du = np.sqrt(du_dx**2 + du_dy**2)
    laplace_u = d['laplacian_matrix'].dot(_sol_flat)
    # residual = laplace_u - d['F']
    # logger.info('residual', np.linalg.norm(residual))
    du_dx = -du_dx.reshape((Nx, Ny), order='F')
    du_dy = -du_dy.reshape((Nx, Ny), order='F')
    du = du.reshape((Nx, Ny), order='F')
    laplace_u = laplace_u.reshape((Nx, Ny), order='F')
    # residual = residual.reshape((Nx, Ny), order='F')
    logger.info("Integral: {}".format(integrateSolution(Nx, Ny, sol, d['Jacobian'])))
    plotSolution(Nx, Ny, sol, transformation=transformation, contour_levels=contour_levels, gradient=(du, du_dx, du_dy), laplace_u=laplace_u)
