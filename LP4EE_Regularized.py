import numpy as np
from scipy import sparse
import osqp

class LP4EE_Regularized:

    def __init__(self, regularization='L1', alpha=1.0, l1_ratio=0.5):

        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.upper_bound = 1e16
        self.coefficients_ = None

    def fit(self, X, y):
        self.coefficients_ = self._fit_reg_lp(X, y)
        return self

    def predict(self, X):
        if self.coefficients_ is None:
            raise ValueError("Model must be fitted before making predictions")
        else:
            pred_y = np.dot(X, self.coefficients_)
        return pred_y

    def _fit_reg_lp(self, train_X, train_y):

        X = train_X.astype(np.float64, copy=False)
        y = train_y.astype(np.float64, copy=False)

        n_samples, n_features = X.shape
        n_variables = n_features + 2 * n_samples

        if self.regularization == 'L1':
            l1_penalty = self.alpha
            l2_penalty = 0.0
        elif self.regularization == 'L2':
            l1_penalty = 0.0
            l2_penalty = self.alpha
        elif self.regularization.startswith('EN'):
            l1_penalty = self.alpha * self.l1_ratio
            l2_penalty = self.alpha * (1 - self.l1_ratio)
        else:
            raise ValueError("Penalty must be one of 'L1', 'L2', or 'EN'")

        diagonal = np.zeros(n_variables)
        diagonal[:n_features] = 2 * l2_penalty + 1e-12
        diagonal[n_features:] = 1e-12
        P = sparse.diags(diagonal, format='csc')

        q = np.zeros(n_variables, dtype=np.float64)
        q[:n_features] = l1_penalty
        q[n_features:] = 1.0

        A_eq = sparse.hstack([
            X,
            -sparse.eye(n_samples),
            sparse.eye(n_samples)
        ], format='csc')

        A = sparse.vstack([A_eq, sparse.eye(n_variables)], format='csc')

        l_bounds = np.zeros(n_variables)
        u_bounds = np.full(n_variables, np.inf)
        u_bounds[:n_features] = self.upper_bound

        l = np.hstack([y, l_bounds])
        u = np.hstack([y, u_bounds])

        solver = osqp.OSQP()
        for eps_rel in [1e-3, 1e-2]:
            solver.setup(P=P, q=q, A=A, l=l, u=u,
                         verbose=False,
                         eps_rel=eps_rel,
                         max_iter=100_000)
            result = solver.solve()
            if result.info.status == 'solved':
                coeffs = result.x[:n_features]
                return np.maximum(coeffs, 0)

        raise RuntimeError(f"Optimization failed")