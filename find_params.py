
from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import optuna
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import torch
from optuna.samplers import RandomSampler, CmaEsSampler, TPESampler
from tsallis import Tsallian, pirsonian, simple_tsallis_, simple_tsallis_torch


def find_param(
    hm,
    func,
    initial_ampl,  # параметры начальной точки
    initial_qM,
    inital_G,
    tsal_cropped,
    bounds  # искаженный обрезанный тцаллиан
):

    # Задаем начальную точку: ampl, q(или M в случае пирсониана), G
    initial_guess = [
        initial_ampl,
        initial_qM,
        inital_G
    ]

    # Создаем шаблон функции, в который поместим пирсониан и/или тцаллиан в зависимости от мода
    params, cov = curve_fit(
        func,
        tsal_cropped.B,
        tsal_cropped.Y_norm,
        p0=initial_guess,
        method='trf',
        bounds=bounds
    )

    amplt, qt, Gt = params

    fitted_y = func(tsal_cropped.B, amplt, qt, Gt)

    msn = np.mean((tsal_cropped.Y_norm - fitted_y) ** 2)

    type_func = "пирсониана" if func == pirsonian else "тцаллиана"

    print(f"Параметры {type_func}:\n"
          f"hm={hm:.3f}, qt={qt:.4f}, Gt={Gt:.2f}, amplt={amplt:.2f}, msn={msn:.8f}")

    return params, msn


def quadratic_error(variables, *args):
    G1, q1, Bres1, Ampl1 = variables
    x = args[0]
    y = args[1]
    y_pred = simple_tsallis_(x, q1, G1, Bres1, Ampl1, 0)
    return np.sum((y - y_pred) ** 2)/np.size(y_pred)


def quadratic_error_c(variables, *args):
    G1, q1, Bres1, Ampl1, c = variables
    x = args[0]
    y = args[1]
    y_pred = simple_tsallis_(x, q1, G1, Bres1, Ampl1, c)
    return np.sum((y - y_pred) ** 2)/np.size(y_pred)


def quadratic_error_bfix(variables, *args):
    G1, q1, Ampl1 = variables
    Bres1 = 3250
    x = args[0]
    y = args[1]
    a1 = pow(2.0, q1 - 1.0) - 1.0
    a2 = -1.0 / (q1 - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - Bres1) / G1, 2.0), a2 - 1.0) * a1 * ((x - Bres1) / np.power(G1, 2))
    f_max = np.max(SS)
    St = Ampl1 * (SS / (2.0 * f_max))
    return np.sum((y - St) ** 2)/np.size(St)


def quadratic_error_bfix(variables, *args):
    G1, q1, Ampl1 = variables
    Bres1 = 3250
    x = args[0]
    y = args[1]
    a1 = pow(2.0, q1 - 1.0) - 1.0
    a2 = -1.0 / (q1 - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - Bres1) / G1, 2.0), a2 - 1.0) * a1 * ((x - Bres1) / np.power(G1, 2))
    f_max = np.max(SS)
    St = Ampl1 * (SS / (2.0 * f_max))
    return np.sum((y - St) ** 2)/np.size(St)


def quadratic_error_bfix_ampl(variables, *args):
    G1, q1 = variables
    Bres1 = 3250
    x = args[0]
    y = args[1]
    Ampl1 = args[2]
    a1 = pow(2.0, q1 - 1.0) - 1.0
    a2 = -1.0 / (q1 - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - Bres1) / G1, 2.0), a2 - 1.0) * a1 * ((x - Bres1) / np.power(G1, 2))
    f_max = np.max(SS)
    St = Ampl1 * (SS / (2.0 * f_max))
    return np.sum((y - St) ** 2)/np.size(St)


def quadratic_error_bfix_lm(variables, x, y):
    G1 = variables['G']
    q1 = variables['q']
    Ampl1 = variables['Ampl']
    Bres1 = 3250
    a1 = pow(2.0, q1 - 1.0) - 1.0
    a2 = -1.0 / (q1 - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - Bres1) / G1, 2.0), a2 - 1.0) * a1 * ((x - Bres1) / np.power(G1, 2))
    f_max = np.max(SS)
    St = Ampl1 * (SS / (2.0 * f_max))
    return y - St


dBpp_theor = 0
dBpp_exper = 0


def quadratic_error_b(variables, *args):
    Bres1 = variables
    X_list = args[0]
    Y_list = args[1]
    ma = args[2]
    Ampl0_1 = 0.990691551703254
    C = 0.00019618396712957776

    params_1 = {
        "q0": 2.131899242280023,
        "G0": 0.5620619750350033,
        "B0": Bres1,
        "H_array": X_list,
        "hm": ma,
    }
    Y1 = Tsallian().tsall_init_new(*list(params_1.values()))
    Y_sum = Y1*Ampl0_1 + C
    global dBpp_theor, dBpp_exper
    dBpp_theor = X_list[np.argmin(Y1)] - X_list[np.argmax(Y1)]
    dBpp_exper = X_list[np.argmin(Y_list)] - X_list[np.argmax(Y_list)]
    funmin = np.sum((Y_sum-Y_list)**2)/len(Y_sum)
    return funmin


def quadratic_error_two(variables, *args):
    G1, q1, Bres1, Ampl1, G2, q2, Bres2, Ampl2, c = variables
    x = args[0]
    y = args[1]
    y_pred = simple_tsallis_(x, q1, G1, Bres1, Ampl1, 0) + simple_tsallis_(x, q2, G2, Bres2, Ampl2, 0) + c
    return np.sum((y - y_pred) ** 2)/np.size(y_pred)


def objective_one(trial, x_data, y_data):
    B_0 = trial.suggest_float("B1", 3249, 3251)
    G_0 = 2
    q_0 = 2
    Ampl_0 = 0.7
    dG = 1
    dq = 0.99999
    dB = 0.5
    dAmp = 0.49999
    c = 0
    dc = 0.2

    initial_guess = [G_0, q_0, B_0, Ampl_0, c]
    bounds = [
        (G_0-dG, G_0+dG), (q_0-dq, q_0+dq),
        (B_0-dB, B_0+dB), (Ampl_0-dAmp, Ampl_0+dAmp),
        (c-dc, c+dc)
    ]

    res = minimize(
        quadratic_error_c,
        initial_guess,
        args=(x_data, y_data),
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'maxfun': 500000,
            'maxiter': 50000000,
            'ftol': 1e-6
        }
    )

    if len(trial.study.best_trials):
        if res.fun < trial.study.best_value:
            trial.set_user_attr("res", res)
    return res.fun


def objective_two(trial, x_data, y_data):
    B_0 = trial.suggest_float("B1", 3250, 3261)
    B_1 = trial.suggest_float("B2", 3250, 3261)
    G_0 = 2
    q_0 = 2
    Ampl_0 = 0.5
    dG = 1
    dq = 0.99999
    dB = 0.5
    dAmp = 0.49999

    initial_guess = [G_0, q_0, B_0, Ampl_0]
    bounds = [
        (G_0-dG, G_0+dG), (q_0-dq, q_0+dq),
        (B_0-dB, B_0+dB), (Ampl_0-dAmp, Ampl_0+dAmp)
    ]

    res = minimize(
        quadratic_error,
        initial_guess,
        args=(x_data, y_data),
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
    dB = 0.5
    dAmp = 0.4999999
    c = 0
    dc = 0.5
    initial_guess = [res.x[0], res.x[1], res.x[2], res.x[3], G_0, q_0, B_1, Ampl_0, c]
    bounds = [
        (res.x[0]-dG, res.x[0]+dG), (res.x[1]-dq, res.x[1]+dq),
        (res.x[2]-dB, res.x[2]+dB), (res.x[3]-dAmp, res.x[3]+dAmp),
        (G_0-dG, G_0+dG), (q_0-dq, q_0+dq),
        (B_1-dB, B_1+dB), (Ampl_0-dAmp, Ampl_0+dAmp),
        (c-dc, c+dc)
    ]

    res = minimize(
        quadratic_error_two,
        initial_guess,
        args=(x_data, y_data),
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'maxfun': 500000,
            'maxiter': 50000000,
            'ftol': 1e-6
        }
    )
    if len(trial.study.best_trials):
        if res.fun < trial.study.best_value:
            trial.set_user_attr("res", res)
    return res.fun


def objective_one_b(trial, x_data, y_data, ma):
    dB = 0.5
    B_0 = trial.suggest_float("B1", 3252.91258671749-dB, 3252.91258671749+dB)
    initial_guess = [B_0]
    bounds = [
        (B_0-dB, B_0+dB)
    ]

    res = minimize(
        quadratic_error_b,
        initial_guess,
        args=(x_data, y_data, ma),
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'maxfun': 500000,
            'maxiter': 50000000,
            'ftol': 1e-6
        }
    )

    if len(trial.study.best_trials):
        if res.fun < trial.study.best_value:
            trial.set_user_attr("res", res)
    return res.fun


opt = 1000000
q1opt = None,
G1opt = None,
Bres1opt = None,
Ampl1opt = None,
q2opt = None,
G2opt = None,
Bres2opt = None,
Ampl2opt = None,
copt = None


def objective(trial, x_data, y_data):
    G1 = trial.suggest_float("G1", 0, 4)
    q1 = trial.suggest_float("q1", 1, 3)
    Bres1 = trial.suggest_float("Bres1", 3245, 3260)
    Ampl1 = trial.suggest_float("Ampl1", 0, 1)
    G2 = trial.suggest_float("G2", 0, 4)
    q2 = trial.suggest_float("q2", 1, 3)
    Bres2 = trial.suggest_float("Bres2", 3245, 3260)
    Ampl2 = trial.suggest_float("Ampl2", 0, 1)
    c = trial.suggest_float("c", -10, 10)
    predefined_values = [q1, G1, Bres1, Ampl1, q2, G2, Bres2, Ampl2, c]
    return find_loss(torch.from_numpy(x_data.values), torch.from_numpy(y_data.values), predefined_values)


def check_stop(study, trial):
    if study.best_value < 2.5e-6:
        study.stop()


def find_loss(x_data, y_data, predefined_values):

    q1, G1, Bres1, Ampl1, q2, G2, Bres2, Ampl2, c = [
        torch.tensor([value], dtype=torch.float32, requires_grad=True) for value in predefined_values
    ]
    global opt
    global q1opt, G1opt, Bres1opt, Ampl1opt, q2opt, G2opt, Bres2opt, Ampl2opt, copt
    optimizer = torch.optim.SGD([q1, G1, Bres1, Ampl1, q2, G2, Bres2, Ampl2, c], lr=0.01)
    # Цикл обучения
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = simple_tsallis_torch(x_data, q1, G1, Bres1, Ampl1, 0) + \
            simple_tsallis_torch(x_data, q2, G2, Bres2, Ampl2, 0) + c
        output_new = torch.sum((y_data - y_pred) ** 2) / y_pred.numel()
        if output_new < opt:
            q1opt, G1opt, Bres1opt, Ampl1opt, q2opt, G2opt, Bres2opt, Ampl2opt, copt = \
                q1.item(), G1.item(), Bres1.item(), Ampl1.item(), \
                q2.item(), G2.item(), Bres2.item(), Ampl2.item(), c.item()
            opt = output_new
            print(opt, "\n",
                  q1opt, G1opt, Bres1opt, Ampl1opt, q2opt, G2opt, Bres2opt, Ampl2opt, copt)
        output_new.backward()
        optimizer.step()

    return opt


def find_one_tsall_param(
    experimental_spectr
):
    X_list = experimental_spectr['B']
    Y_list = experimental_spectr['Signal']/(max(experimental_spectr['Signal']) - min(experimental_spectr['Signal']))
    objective_with_data = partial(objective_one, x_data=X_list, y_data=Y_list)
    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    study.optimize(objective_with_data, n_trials=100, callbacks=[check_stop])
    return study.best_trial.user_attrs['res']


def find_two_tsall_param(
    experimental_spectr
):
    X_list = experimental_spectr['B']
    Y_list = experimental_spectr['Signal']/(max(experimental_spectr['Signal']) - min(experimental_spectr['Signal']))
    objective_with_data = partial(objective_two, x_data=X_list, y_data=Y_list)
    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    study.optimize(objective_with_data, n_trials=100, callbacks=[check_stop])

    # objective_with_data = partial(objective, x_data=X_list, y_data=Y_list)
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective_with_data, n_trials=5000, callbacks=[check_stop])

    # plt.figure(figsize=(8, 5))
    # plt.scatter(X_list, Y_list, label='Исходные данные')
    # plt.scatter(
    #     X_list,
    #     simple_tsallis_(
    #                 X_list, q2opt, G2opt,
    #                 Bres2opt, Ampl2opt, 0),
    #     label=''
    # )
    # plt.text(3200, -80, f"Параметры подгонки:\nq2={q2opt}\nG2opt={G2opt}"
    #          f"\nB2={Bres2opt}\nAmpl2opt={Ampl2opt}")

    # plt.figure(figsize=(8, 5))
    # plt.scatter(X_list, Y_list, label='Исходные данные')
    # plt.scatter(
    #     X_list,
    #     simple_tsallis_(
    #                 X_list, q1opt, G1opt,
    #                 Bres1opt, Ampl1opt, 0),
    #     label=''
    # )
    # plt.text(3200, -80, f"Параметры подгонки:\nq1={q1opt}\nG1opt={G1opt}"
    #          f"\nB1={Bres1opt}\nAmpl1opt={Ampl1opt}")

    # plt.figure(figsize=(8, 5))
    # plt.scatter(X_list, Y_list, label='Исходные данные')
    # plt.scatter(
    #     X_list,
    #     simple_tsallis_(
    #                 X_list, q1opt, G1opt,
    #                 Bres1opt, Ampl1opt, 0) +
    #     simple_tsallis_(
    #                 X_list, q2opt, G2opt,
    #                 Bres1opt, Ampl2opt, 0) + copt,
    #     label=''
    # )
    # print("Лучшие параметры:", study.best_params)

    # G_0 = 2
    # q_0 = 2
    # B_0 = 3250
    # Ampl_0 = 100
    # dG = 1
    # dq = 1
    # dB = 10
    # dAmp = 50
    # c = 1
    # initial_guess = [G_0, q_0, B_0, Ampl_0, G_0, q_0, B_0, Ampl_0, c]
    # bounds = [
    #     (G_0-dG, G_0+dG), (q_0-dq, q_0+dq),
    #     (B_0-dB, B_0+dB), (Ampl_0-dAmp, Ampl_0+dAmp),
    #     (G_0-dG, G_0+dG), (q_0-dq, q_0+dq),
    #     (B_0-dB, B_0+dB), (Ampl_0-dAmp, Ampl_0+dAmp),
    #     (c-10, c+10)]

    # res = minimize(
    #     quadratic_error,
    #     initial_guess,
    #     args=(X_list, Y_list),
    #     bounds=bounds,
    #     options={
    #         'maxfun': 500000,
    #         'maxiter': 50000000,
    #         'ftol': 1e-6
    #     }
    # )

    # plt.figure(figsize=(8, 5))
    # plt.scatter(X_list, Y_list, label='Исходные данные')
    # plt.scatter(X_list, simple_tsallis_(X_list, res.x[1], res.x[0], res.x[2], res.x[3], 0) +
    #             simple_tsallis_(X_list, res.x[5], res.x[4], res.x[6], res.x[7], 0) + res.x[8], label='')
    # study.best_trial.user_attrs['res']
    return None


def find_one_tsall_param_b(
    X_list,
    Y_list,
    ma
):
    objective_with_data = partial(objective_one_b, x_data=X_list, y_data=Y_list, ma=ma)
    study = optuna.create_study(direction="minimize", sampler=TPESampler())
    study.optimize(objective_with_data, n_trials=3, callbacks=[check_stop])
    res = study.best_value
    return res
