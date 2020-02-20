import tensorflow as tf
import numpy as np
from tqdm import tqdm
try:
    import quadprog
except ImportError:
    import warnings
    warnings.warn("not able to import quadprog."
                  " the successive convex optimizer wont work.")

__all__ = ['SuccessiveConvexOptimizer']


class SuccessiveConvexOptimizerValidator:

    def __init__(self):
        pass

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if not (0 < value < 1):
            raise ValueError("gamma has to be a real number in the interval"
                             "(0, 1), got {}.".format(value))
        else:
            try:
                self._gamma = float(value)
            except Exception as e:
                raise e

    @property
    def zeta(self):
        return self._zeta

    @zeta.setter
    def zeta(self, value):
        if value < 0:
            raise ValueError("zeta has to be a positive real number, got {}.".format(value))
        else:
            try:
                self._zeta = float(value)
            except Exception as e:
                raise e

    @property
    def funtol(self):
        return self._funtol

    @funtol.setter
    def funtol(self, value):
        if value < 0:
            raise ValueError("funtol has to be a positive real number, got {}.".format(value))
        else:
            try:
                self._funtol = float(value)
            except Exception as e:
                raise e

    @property
    def wtol(self):
        return self._wtol

    @wtol.setter
    def wtol(self, value):
        if value < 0:
            raise ValueError("wtol has to be a positive real number, got {}.".format(value))
        else:
            try:
                self._wtol = float(value)
            except Exception as e:
                raise e
    @property
    def maxiter(self):
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        if value < 0:
            raise ValueError("maxiter has to be a positive integer, got {}.".format(value))
        else:
            try:
                self._maxiter = int(value)
            except Exception as e:
                raise e


class SuccessiveConvexOptimizer:
    """
    Successive Convex Approximation optimizer tailored for the risk parity problem.
    """
    def __init__(self, portfolio, tau = None, gamma = 0.9, zeta = 1E-7,
                 funtol = 1E-6, wtol = 1E-6, maxiter = 500, Cmat = None,
                 cvec = None, Dmat = None, dvec = None):
        self.portfolio = portfolio
        self.tau       = (tau or 0.05 * np.sum(np.diag(self.portfolio.covariance.numpy()))
                                 / (2 * self.portfolio.number_of_assets))
        sca_validator  = SuccessiveConvexOptimizerValidator()
        self.gamma     = sca_validator.gamma     = gamma
        self.zeta      = sca_validator.zeta      = zeta
        self.funtol    = sca_validator.funtol    = funtol
        self.wtol      = sca_validator.wtol      = wtol
        self.maxiter   = sca_validator.maxiter   = maxiter
        self.Cmat      = Cmat
        self.Dmat      = Dmat
        self.cvec      = cvec
        self.dvec      = dvec
        self.CCmat     = np.vstack((self.Cmat, self.Dmat)).T
        self.bvec      = np.concatenate((self.cvec, self.dvec))
        self.meq       = self.Cmat.shape[0]
        self._funk = self.get_objective_function_value()
        self.objective_function = [self._funk.numpy()]
        self._tauI = self.tau * tf.eye(self.portfolio.number_of_assets, dtype=tf.float64)

    @property
    def Cmat(self):
        return self._Cmat

    @Cmat.setter
    def Cmat(self, value):
        if value is None:
            self._Cmat = np.atleast_2d(np.ones(self.portfolio.number_of_assets))
        elif np.atleast_2d(value).shape[1] == self.portfolio.number_of_assets:
            self._Cmat = np.atleast_2d(value)
        else:
            raise ValueError("Cmat shape {} doesnt agree with the number of"
                             "assets {}".format(value.shape, self.number_of_assets))

    @property
    def Dmat(self):
        return self._Dmat

    @Dmat.setter
    def Dmat(self, value):
        if value is None:
            self._Dmat = np.eye(self.portfolio.number_of_assets)
        elif np.atleast_2d(value).shape[1] == self.portfolio.number_of_assets:
            self._Dmat = -np.atleast_2d(value)
        else:
            raise ValueError("Dmat shape {} doesnt agree with the number of"
                             "assets {}".format(value.shape, self.number_of_assets))

    @property
    def cvec(self):
        return self._cvec

    @cvec.setter
    def cvec(self, value):
        if value is None:
            self._cvec = np.array([1])
        elif len(value) == self.Cmat.shape[0]:
            self._cvec = value
        else:
            raise ValueError("cvec shape {} doesnt agree with Cmat shape"
                             "{}".format(value.shape, self.Cmat.shape))

    @property
    def dvec(self):
        return self._dvec

    @dvec.setter
    def dvec(self, value):
        if value is None:
            self._dvec = np.zeros(self.portfolio.number_of_assets)
        elif len(value) == self.Dmat.shape[0]:
            self._dvec = -value
        else:
            raise ValueError("dvec shape {} doesnt agree with Dmat shape"
                             "{}".format(value.shape, self.Dmat.shape))

    def get_objective_function_value(self):
        obj = self.portfolio.risk_concentration.evaluate()
        if self.portfolio.has_mean_return:
            obj -= self.portfolio.alpha * self.portfolio.mean_return
        if self.portfolio.has_variance:
            obj += self.portfolio.lmd * self.portfolio.variance
        return obj

    def iterate(self):
        wk = self.portfolio.weights
        g = self.portfolio.risk_concentration.risk_concentration_vector()
        A = self.portfolio.risk_concentration.jacobian_risk_concentration_vector()
        At = tf.transpose(A)
        Q = 2 * At @ A + self._tauI
        q = 2 * tf.linalg.matvec(At, g) - tf.linalg.matvec(Q, wk)
        if self.portfolio.has_variance:
            Q += self.portfolio.lmd * self.portfolio.covariance
        if self.portfolio.has_mean_return:
            q -= self.portfolio.alpha * self.portfolio.mean
        w_hat = quadprog.solve_qp(Q.numpy(), -q.numpy(), C=self.CCmat, b=self.bvec, meq=self.meq)[0]
        #w_hat = qp.solve(Qmat=Q.numpy(), qvec=q.numpy(), Cmat=self.Cmat, cvec=self.cvec,
        #                 Dmat=self.Dmat, dvec=self.dvec, w0=wk, maxiter=self.maxiter,
        #                 tol=self.wtol)
        self.portfolio.weights = wk + self.gamma * (w_hat - wk)
        fun_next = self.get_objective_function_value()
        self.objective_function.append(fun_next.numpy())
        has_w_converged = (tf.abs(self.portfolio.weights - wk) <=
                           .5 * self.wtol * (tf.abs(self.portfolio.weights) +
                                             tf.abs(wk))).numpy().all()
        has_fun_converged = (tf.abs(self._funk - fun_next) <=
                             .5 * self.funtol * (tf.abs(self._funk) +
                                                 tf.abs(fun_next))).numpy().all()
        if has_w_converged or has_fun_converged:
            return False
        self.gamma = self.gamma * (1 - self.zeta * self.gamma)
        self._funk = fun_next
        return True

    def solve(self):
        i = 0
        with tqdm(total=self.maxiter) as pbar:
            while(self.iterate() and i < self.maxiter):
                i += 1
                pbar.update()


def project_line_and_box(weights, lower_bound, upper_bound):
    def objective_function(variable, weights):
        pass
    pass
