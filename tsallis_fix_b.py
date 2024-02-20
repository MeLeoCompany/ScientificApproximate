import numpy as np
from scipy.optimize import minimize

from tsallis import simple_tsallis_


def quadratic_error_fixB(variables, *args):
    G1, q1, Ampl1 = variables
    x = args[0]
    y = args[1]
    Bres = args[2]
    y_pred = simple_tsallis_(x, q1, G1, Bres, Ampl1, 0)
    return np.sum((y - y_pred) ** 2)/np.size(y_pred)


def quadratic_error_two_fixB(variables, *args):
    G1, q1, Ampl1, G2, q2, Ampl2, c = variables
    x = args[0]
    y = args[1]
    Bres1 = args[2]
    Bres2 = args[3]
    y_pred = simple_tsallis_(x, q1, G1, Bres1, Ampl1, 0) + simple_tsallis_(x, q2, G2, Bres2, Ampl2, 0) + c
    return np.sum((y - y_pred) ** 2)/np.size(y_pred)


def find_params_two_tsallis_fixB(x_data, y_data, B_0, B_1):
    G_0 = 2
    q_0 = 2
    Ampl_0 = 0.5
    dG = 1
    dq = 0.99999
    dAmp = 0.49999

    initial_guess = [G_0, q_0, Ampl_0]
    bounds = [
        (G_0-dG, G_0+dG), (q_0-dq, q_0+dq),
        (Ampl_0-dAmp, Ampl_0+dAmp)
    ]

    res = minimize(
        quadratic_error_fixB,
        initial_guess,
        args=(x_data, y_data, B_0),
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'maxfun': 500000,
            'maxiter': 50000000,
            'ftol': 1e-6
        }
    )

    G_0 = 2
    q_0 = 2
    Ampl_0 = 0.5
    dG = 1
    dq = 0.999999
    dAmp = 0.4999999
    c = 0
    dc = 0.5
    initial_guess = [res.x[0], res.x[1], res.x[2], G_0, q_0, Ampl_0, c]
    bounds = [
        (res.x[0]-dG, res.x[0]+dG), (res.x[1]-dq, res.x[1]+dq),
        (res.x[2]-dAmp, res.x[2]+dAmp),
        (G_0-dG, G_0+dG), (q_0-dq, q_0+dq),
        (Ampl_0-dAmp, Ampl_0+dAmp),
        (c-dc, c+dc)
    ]

    res = minimize(
        quadratic_error_two_fixB,
        initial_guess,
        args=(x_data, y_data, B_0, B_1),
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'maxfun': 500000,
            'maxiter': 50000000,
            'ftol': 1e-6
        }
    )
    return res
