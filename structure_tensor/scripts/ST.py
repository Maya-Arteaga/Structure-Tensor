from structure_tensor import eig_special_2d, structure_tensor_2d
import ACG
import numpy as np


def cart2pol(x, y):
    R = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    return R, angle


def distribute_around_cercle(x, y):
    if len(x) == len(y):
        rand = np.random.random(len(x))
        for i in range(0, len(rand)):
            if rand[i] < 0.5:
                rand[i] = -1
            if rand[i] >= 0.5:
                rand[i] = 1
        return x * rand, y * rand
    if len(x) != len(y):
        return x, y


def ST(sigma, rho, k, Image):

    # apply gaussian filter to the image
    S = structure_tensor_2d(Image.astype(float), sigma, rho)
    val, vec = eig_special_2d(S)

    # -------------------------------------------------------------------------------------------------
    len0 = len(vec[0, :, 0])
    len1 = len(vec[0, 0, :])
    # remove the borders + orthogonal vector:
    # vec = [a,b], vec_ort = [-b,a]
    vec_ort = np.empty((2, len0-2*k, len1-2*k))
    vec_ort[0] = -vec[1, k:len0-k, k:len1-k]
    vec_ort[1] = vec[0, k:len0-k, k:len1-k]

    # vec_cut = np.empty((2, len0-2*k, len1-2*k))
    # vec_cut[0] = vec[0, k:len0-k, k:len1-k]
    # vec_cut[1] = vec[1, k:len0-k, k:len1-k]
    # vec_ort = vec_cut.copy()
    # vec_ort[0, :, :] = -vec_cut[1, :, :]
    # vec_ort[1, :, :] = vec_cut[0, :, :]
    # -------------------------------------------------------------------------------------------------

    [vec_ort[0, :, :], vec_ort[1, :, :]] = distribute_around_cercle(vec_ort[0, :, :], vec_ort[1, :, :])

    smallN = np.zeros((np.shape(vec_ort)[1], np.shape(vec_ort)[2]))
    smallN[:, :] = cart2pol(vec_ort[0, :, :], vec_ort[1, :, :])[1]

    # for the ACG:
    xN = vec_ort[0, :, :].flatten()
    yN = vec_ort[1, :, :].flatten()
    ACG_N = np.vstack((xN, yN))  # a two dim array with coord x at first row and coord y at second row

    # ACG of the normal component
    pp = ACG.estimate_parameters(ACG_N, np.shape(ACG_N)[0])
    # vaps = eigenvalues; veps = eigenvectors
    if np.linalg.det(pp) != 0:
        vaps, veps = np.linalg.eig(np.linalg.inv(pp))  # will be used later for the plot
    else:
        vaps = np.zeros(2)
        veps = 0

    rad = np.linspace(-np.pi, np.pi, num=60)

    ff = ACG.function(np.shape(ACG_N)[0], rad, pp)
    F = np.array([ff * np.cos(rad), ff * np.sin(rad)])
    # print(np.shape(ff))
    return F, vaps, veps
