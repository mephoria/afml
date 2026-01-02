import numpy as np


#I didnt really get what this does, come back to this later
def pca_weights(cov, risk_distribution=None, risk_target=1):
    eigenvalue, eigenvector = np.linalg.eigh(cov)
    indices = eigenvalue.argsort()[::-1]
    eigenvalue, eigenvector = eigenvalue[indices], eigenvector[:, indices]

    if risk_distribution is None:
        risk_distribution = np.zeros(cov.shape[0])
        risk_distribution[-1] = 1

    loads = risk_target * (risk_distribution / eigenvalue) ** 0.5
    weights = np.dot(eigenvector, np.reshape(loads, (-1, 1)))

    return weights