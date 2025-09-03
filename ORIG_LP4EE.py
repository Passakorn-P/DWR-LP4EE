import numpy as np
from scipy.optimize import linprog

class ORIG_LP4EE:

    @staticmethod
    def predict(train_X, train_y, test_X):

        n_samples, n_features = train_X.shape
        n_variables = n_features + 2 * n_samples

        c = np.zeros(n_variables)
        tzip_indices = n_features + np.arange(n_samples) * 2 - 1
        c[tzip_indices] = 1

        mat1 = np.zeros((n_samples, n_variables))
        mat1[:, :n_features] = train_X

        mat2 = mat1.copy()
        for i in range(n_samples):
            mat2[i, n_features + i * 2 - 1] = -1
            mat2[i, n_features + i * 2] = -1

        mat3 = mat2.copy()
        for i in range(n_samples):
            mat3[i, n_features + i * 2 - 1] = 1

        mat4 = np.zeros((n_features, n_variables))
        for i in range(n_features):
            mat4[i, i] = 1

        mat5 = mat4.copy()

        mat6 = np.zeros((n_samples, n_variables))
        for i in range(n_samples):
            mat6[i, n_features + i * 2] = 1

        A_ub = np.vstack([-mat1, mat2, -mat3, -mat4, mat5])
        b_ub = np.concatenate([
            np.zeros(n_samples),
            np.zeros(n_samples),
            np.zeros(n_samples),
            np.zeros(n_features),
            np.ones(n_features) * 1e16
        ])

        A_eq = mat6
        b_eq = train_y

        bounds = [(0, None) for _ in range(n_variables)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            coefficients = result.x[:n_features]

            predictions = np.dot(test_X, coefficients)

            return predictions, coefficients
        else:
            raise RuntimeError(f"Optimization failed: {result.message}")
