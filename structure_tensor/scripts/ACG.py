# Angular central gaussian distribution, Tyler 1987

# A should be symmetric and positive definite
# if q = dim; a = 2 pi^{q/2} / gamma(q/2)

# L(x,A) = 1 / ((a * |A|^{1/2}) * (x^{T} A^{-1} x)^{q/2})

import numpy as np
from scipy.special import gamma


def estimate_parameters(dx, dim):
    A = np.identity(dim)
    A0 = np.zeros([dim, dim])
    # this while takes a lot of time
    while np.sum(np.square(A - A0)) > 0.00001:
        s = 0
        M = np.zeros([dim, dim])
        if np.linalg.det(A) != 0:
            invA = np.linalg.inv(A)
        else:
            return A    # not sure about the consequences

        for i in range(np.shape(dx)[1]):
            x = np.array([dx[:, i]]).T  # select per columns
            z = np.transpose(x) @ invA @ x
            M = M + x @ np.transpose(x) / z
            s = s + 1 / z
        A0 = A
        A = dim * M / s

    return A


def function(dim, angles, M):
    t = np.empty([2, np.size(angles)])
    t[0] = np.cos(angles)
    t[1] = np.sin(angles)

    a = 2 * np.pi ** (dim / 2) / gamma(dim / 2)
    k = a * np.linalg.det(M) ** (1 / 2)
    lml = []
    if np.linalg.det(M) != 0:
        for i in range(np.shape(t)[1]):
            lml.append((np.transpose(t[:, i]) @ np.linalg.inv(M) @ t[:, i]) ** (dim / 2))

        f = [1 / (k * i) for i in lml]
    else:
        f = 0

    return f
