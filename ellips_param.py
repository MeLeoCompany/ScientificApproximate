import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def quadratic_error(variables, *args):
    p2,p3,p4 = variables # p2 примерно равняетя 1/M = q-1 в относительных, p3 от 0 до 2, p4 от -0.5 до 0.5 
    x = args[0]
    y = args[1]
    y_pred = 0+p2*(1-((x-p4)/p3)**2)**0.5
    return np.sum((y - y_pred) ** 2)/np.size(y_pred)

def find_ellips_param(
        q0,
        X_list, 
        Y_list
    ):

    p4_list = np.linspace(-0.5, 0.5, 4)
    dp4 = p4_list[1] - p4_list[0]
    p3_list = np.linspace(0, 2, 4)
    dp3 = p3_list[1] - p3_list[0]
    p2_list = np.linspace(q0-0.5, q0+0.5, 4)
    dp2 = p2_list[1] - p2_list[0]
    centers = [
    (
        (p2_list[i] + p2_list[i+1]) / 2,
        (p3_list[j] + p3_list[j+1]) / 2,
        (p4_list[k] + p4_list[k+1]) / 2
    )
    for i in range(len(p2_list) - 1)
    for j in range(len(p3_list) - 1)
    for k in range(len(p4_list) - 1)
    ]

    success = False
    minfun = 1000000
    for p2, p3, p4 in centers:
        initial_guess = [p2, p3, p4]  # Начальное приближение для параметров a и b
        bounds = [(p2-dp2, p2+dp2), (p3-dp3, p3+dp3), (p4-dp4, p4+dp4)]
        if all(1-((X_list-dp4)/p3)**2 >= 0):
            result = minimize(
                quadratic_error, 
                initial_guess, 
                method='L-BFGS-B', 
                args=(X_list, Y_list), 
                bounds=bounds,
                options={
                    'maxfun': 500000, 
                    'maxiter': 50000000, 
                }
            )
            if result.success == True and  result.fun != 0:
                if minfun > result.fun:
                    minfun = result.fun
                    orimize_p2, orimize_p3, orimize_p4 = result.x[0], result.x[1], result.x[2]
                success = True
        else:
            continue
    
    if success == True:
        print(f'Для M={q0} найдено решение аппроксимации эллипсоидом: параметры \n'
              f'{orimize_p2, orimize_p3, orimize_p4}\n')
        # сохраняем график
        plt.figure(figsize=(8, 5))
        plt.scatter(X_list, Y_list, label='Исходные данные')
        plt.scatter(X_list, result.x[0]*(1-((X_list-result.x[2])/result.x[1])**2)**0.5, label='Эллипсоид')
        plt.title(f'Аппроксимация эллипсоидом зависимости 1/M от hm/dHpp при q={q0}')
        plt.text(
            min(X_list)+(max(X_list)-min(X_list))/10, 
            min(Y_list)+(max(Y_list)-min(Y_list))/3, 
            f"Параметры подгоночного эллипсоида:\nP2: {result.x[0]:.3f}\nP3: "
            f"{result.x[1]:.3f}\nP4: {result.x[2]:.3f}\n"
            f"Ошибка аппроксимации:\n{result.fun:.6e}", ha='left', va='top')
        plt.xlabel('hm/dHpp')
        plt.ylabel('1/M')
        plt.legend()
        plt.savefig(f'graphs_ellipsoid(q)/ellips_approximate_q={q0}.png')
        plt.close()
        return [orimize_p2, orimize_p3, orimize_p4, result.fun]
    else:
        print(f'Решение для M={q0} не удалось найти')
        return [-1, -1, -1 ,-1]

