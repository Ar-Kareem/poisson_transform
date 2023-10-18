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
            dx_dksi = transformation.dx_dksi(ksi[i, j], eta[i, j])
            dx_deta = transformation.dx_deta(ksi[i, j], eta[i, j])
            dy_dksi = transformation.dy_dksi(ksi[i, j], eta[i, j])
            dy_deta = transformation.dy_deta(ksi[i, j], eta[i, j])
            dx_dksiksi = transformation.dx_dksiksi(ksi[i, j], eta[i, j])
            dx_dksieta = transformation.dx_dksieta(ksi[i, j], eta[i, j])
            dx_detaeta = transformation.dx_detaeta(ksi[i, j], eta[i, j])
            dy_dksiksi = transformation.dy_dksiksi(ksi[i, j], eta[i, j])
            dy_dksieta = transformation.dy_dksieta(ksi[i, j], eta[i, j])
            dy_detaeta = transformation.dy_detaeta(ksi[i, j], eta[i, j])

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
            x_sign = {0: 1, (Nx-1): -1}.get(i, 0)
            y_sign = {0: 1, (Ny-1): -1}.get(j, 0)
            # u_ksi
            if x_sign == 0:  # easy ksi derivative
                Dx_1st_order[NodeMap[i, j], NodeMap[i+1, j]] += +cx_ksi / (2*d_ksi)
                Dx_1st_order[NodeMap[i, j], NodeMap[i-1, j]] += -cx_ksi / (2*d_ksi)
                Dy_1st_order[NodeMap[i, j], NodeMap[i+1, j]] += +cy_ksi / (2*d_ksi)
                Dy_1st_order[NodeMap[i, j], NodeMap[i-1, j]] += -cy_ksi / (2*d_ksi)
            else:  # harder ksi derivative due to ghost point
                # FORWARD: −3f(x) + 4f(x + ∆x) − f(x + 2∆x) /(2∆x)
                # BACKWARD: 3f(x) − 4f(x − ∆x) + f(x − 2∆x) /(2∆x)
                Dx_1st_order[NodeMap[i, j], NodeMap[i, j]]          += -3*x_sign*cx_ksi / (2*d_ksi)
                Dx_1st_order[NodeMap[i, j], NodeMap[i+x_sign, j]]   +=  4*x_sign*cx_ksi / (2*d_ksi)
                Dx_1st_order[NodeMap[i, j], NodeMap[i+2*x_sign, j]] += -1*x_sign*cx_ksi / (2*d_ksi)
                Dy_1st_order[NodeMap[i, j], NodeMap[i, j]]          += -3*x_sign*cy_ksi / (2*d_ksi)
                Dy_1st_order[NodeMap[i, j], NodeMap[i+x_sign, j]]   +=  4*x_sign*cy_ksi / (2*d_ksi)
                Dy_1st_order[NodeMap[i, j], NodeMap[i+2*x_sign, j]] += -1*x_sign*cy_ksi / (2*d_ksi)
            # u_eta
            if y_sign == 0:  # easy eta derivative
                Dx_1st_order[NodeMap[i, j], NodeMap[i, j+1]] += +cx_eta / (2*d_eta)
                Dx_1st_order[NodeMap[i, j], NodeMap[i, j-1]] += -cx_eta / (2*d_eta)
                Dy_1st_order[NodeMap[i, j], NodeMap[i, j+1]] += +cy_eta / (2*d_eta)
                Dy_1st_order[NodeMap[i, j], NodeMap[i, j-1]] += -cy_eta / (2*d_eta)
            else:  # harder eta derivative due to ghost point
                # FORWARD: −3f(x) + 4f(x + ∆x) − f(x + 2∆x) /(2∆x)
                # BACKWARD: 3f(x) − 4f(x − ∆x) + f(x − 2∆x) /(2∆x)
                Dx_1st_order[NodeMap[i, j], NodeMap[i, j]]          += -3*y_sign*cx_eta / (2*d_eta)
                Dx_1st_order[NodeMap[i, j], NodeMap[i, j+y_sign]]   +=  4*y_sign*cx_eta / (2*d_eta)
                Dx_1st_order[NodeMap[i, j], NodeMap[i, j+2*y_sign]] += -1*y_sign*cx_eta / (2*d_eta)
                Dy_1st_order[NodeMap[i, j], NodeMap[i, j]]          += -3*y_sign*cy_eta / (2*d_eta)
                Dy_1st_order[NodeMap[i, j], NodeMap[i, j+y_sign]]   +=  4*y_sign*cy_eta / (2*d_eta)
                Dy_1st_order[NodeMap[i, j], NodeMap[i, j+2*y_sign]] += -1*y_sign*cy_eta / (2*d_eta)

            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:  # skip boundary nodes
                continue

            # INTERIOR POINTS
            # u_ksiksi
            A[NodeMap[i, j], NodeMap[i+1, j  ]] += +c_ksiksi / (d_ksi**2)
            A[NodeMap[i, j], NodeMap[i  , j  ]] += -2*c_ksiksi / (d_ksi**2)
            A[NodeMap[i, j], NodeMap[i-1, j  ]] += +c_ksiksi / (d_ksi**2)
            # u_etaeta
            A[NodeMap[i, j], NodeMap[i  , j+1]] += +c_etaeta / (d_eta**2)
            A[NodeMap[i, j], NodeMap[i  , j  ]] += -2*c_etaeta / (d_eta**2)
            A[NodeMap[i, j], NodeMap[i  , j-1]] += +c_etaeta / (d_eta**2)
            # u_ksieta
            A[NodeMap[i, j], NodeMap[i+1, j+1]] += +c_ksieta / (4*d_ksi*d_eta)
            A[NodeMap[i, j], NodeMap[i+1, j-1]] += -c_ksieta / (4*d_ksi*d_eta)
            A[NodeMap[i, j], NodeMap[i-1, j+1]] += -c_ksieta / (4*d_ksi*d_eta)
            A[NodeMap[i, j], NodeMap[i-1, j-1]] += +c_ksieta / (4*d_ksi*d_eta)
            # u_ksi
            A[NodeMap[i, j], NodeMap[i+1, j  ]] += +c_ksi / (2*d_ksi)
            A[NodeMap[i, j], NodeMap[i-1, j  ]] += -c_ksi / (2*d_ksi)
            # u_eta
            A[NodeMap[i, j], NodeMap[i  , j+1]] += +c_eta / (2*d_eta)
            A[NodeMap[i, j], NodeMap[i  , j-1]] += -c_eta / (2*d_eta)

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
            dx_dksi = transformation.dx_dksi(ksi[i, j], eta[i, j])
            dx_deta = transformation.dx_deta(ksi[i, j], eta[i, j])
            dy_dksi = transformation.dy_dksi(ksi[i, j], eta[i, j])
            dy_deta = transformation.dy_deta(ksi[i, j], eta[i, j])
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
            # β * ∂u/∂n = β* u_x n^x + β* u_y n^y  =  β/J [(y_η n^x − x_η n^y) u_ξ + (-y_ξ n^x + x_ξ n^y) u_η]
            # c_ksi = β/J (y_η n^x-x_η n^y )
            # c_η = β/J (x_ξ n^y-y_ξ n^x )
            c_ksi = beta/J[i, j] * (dy_deta*n__x - dx_deta*n__y)
            c_eta = beta/J[i, j] * (dx_dksi*n__y - dy_dksi*n__x)
            # take derivatives, making sure to not go out of bounds and maintain second order convergence
            x_sign = {0: 1, (Nx-1): -1}.get(i, 0)
            y_sign = {0: 1, (Ny-1): -1}.get(j, 0)
            # u_ksi
            if x_sign == 0:  # easi ksi derivative
                A[NodeMap[i, j], NodeMap[i+1, j]] += +c_ksi / (2*d_ksi)
                A[NodeMap[i, j], NodeMap[i-1, j]] += -c_ksi / (2*d_ksi)
            else:  # harder ksi derivative due to ghost point
                # FORWARD: −3f(x) + 4f(x + ∆x) − f(x + 2∆x) /(2∆x)
                # BACKWARD: 3f(x) − 4f(x − ∆x) + f(x − 2∆x) /(2∆x)
                A[NodeMap[i, j], NodeMap[i, j]]          += -3*x_sign*c_ksi / (2*d_ksi)
                A[NodeMap[i, j], NodeMap[i+x_sign, j]]   +=  4*x_sign*c_ksi / (2*d_ksi)
                A[NodeMap[i, j], NodeMap[i+2*x_sign, j]] += -1*x_sign*c_ksi / (2*d_ksi)

            # u_eta
            if y_sign == 0:  # easy eta derivative
                A[NodeMap[i, j], NodeMap[i, j+1]] += +c_eta / (2*d_eta)
                A[NodeMap[i, j], NodeMap[i, j-1]] += -c_eta / (2*d_eta)
            else:  # harder eta derivative due to ghost point
                # FORWARD: −3f(x) + 4f(x + ∆x) − f(x + 2∆x) /(2∆x)
                # BACKWARD: 3f(x) − 4f(x − ∆x) + f(x − 2∆x) /(2∆x)
                A[NodeMap[i, j], NodeMap[i, j]]          += -3*y_sign*c_eta / (2*d_eta)
                A[NodeMap[i, j], NodeMap[i, j+y_sign]]   +=  4*y_sign*c_eta / (2*d_eta)
                A[NodeMap[i, j], NodeMap[i, j+2*y_sign]] += -1*y_sign*c_eta / (2*d_eta)

    # impose Dirichlet conditions inside Ω (essentially when we know the value of u at certain points)
    laplacian_matrix = A.copy()
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
    return {'A': A, 'b_rhs': b_rhs, 'laplacian_matrix': laplacian_matrix, 'F': F, 'Dx_1st_order': Dx_1st_order, 'Dy_1st_order': Dy_1st_order}

def solveLinear(Nx, Ny, A, b_rhs):
    solution = scipy.sparse.linalg.spsolve(A.tocsr(), b_rhs)
    return solution.reshape((Nx, Ny), order='F')

def integrateSolution(Nx, Ny, solution, transformation=None):
    if transformation is None:
        transformation = get_identity_transformation()
    d_ksi, d_eta = 1/(Nx-1), 1/(Ny-1)
    ksi, eta = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
    J = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny): 
            ksi[i, j] = i*d_ksi
            eta[i, j] = j*d_eta
            J[i, j] = transformation.dx_dksi(ksi[i, j], eta[i, j]) * transformation.dy_deta(ksi[i, j], eta[i, j]) \
                        - transformation.dx_deta(ksi[i, j], eta[i, j]) * transformation.dy_dksi(ksi[i, j], eta[i, j])
    sol_integral = 0
    for j in range(Ny-1):
        for i in range(Nx-1):
            sol_integral += np.mean(solution[i:i+2, j:j+2]) * np.mean(J[i:i+2, j:j+2])*d_ksi*d_eta
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


def plotSolution(Nx: int, Ny: int, solution: np.ndarray, transformation: Transformation = None, contour_levels=20, gradient: (np.ndarray, np.ndarray) = None):
    if transformation is None:
        transformation = get_identity_transformation()
    ksi, eta = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), indexing='ij')
    x = np.array([transformation.Tx(ksi_val, eta_val) for ksi_val, eta_val in zip(ksi.flatten(), eta.flatten())]).reshape(ksi.shape)
    y = np.array([transformation.Ty(ksi_val, eta_val) for ksi_val, eta_val in zip(ksi.flatten(), eta.flatten())]).reshape(ksi.shape)
    if gradient is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
    else:
        plt.figure(figsize=(6, 5))
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
        plt.subplot(1, 2, 2)
        plotStreams(x, y, gradient[1], gradient[2], gradient[0])
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

    gu = griddata(np.r_[ px[None,:], py[None,:] ].T, pu, (xi,yi))
    gv = griddata(np.r_[ px[None,:], py[None,:] ].T, pv, (xi,yi))
    gz = griddata(np.r_[ px[None,:], py[None,:] ].T, pz, (xi,yi))
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
    # nabla_u = d['laplacian_matrix'].dot(_sol_flat)
    # residual = nabla_u - d['F']
    # logger.info('residual', np.linalg.norm(residual))
    du_dx = -du_dx.reshape((Nx, Ny), order='F')
    du_dy = -du_dy.reshape((Nx, Ny), order='F')
    du = du.reshape((Nx, Ny), order='F')
    # nabla_u = nabla_u.reshape((Nx, Ny), order='F')
    # residual = residual.reshape((Nx, Ny), order='F')
    logger.info("Integral: {}".format(integrateSolution(Nx, Ny, sol, transformation)))
    plotSolution(Nx, Ny, sol, transformation=transformation, contour_levels=contour_levels, gradient=(du, du_dx, du_dy))
