from functools import partial
import multiprocessing
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from consts import BOUNDS_PIRS, BOUNDS_TSAL, INITAL_AMPL_PIRS, \
    INITAL_AMPL_TSAL, INITAL_G_PIRS, INITAL_G_TSAL, INITAL_M_PIRS,\
    INITAL_Q_TSAL, ITERATION_DEPTH, PIRSONIAN_MODE, TSALLIAN_MODE

from derivative import derivatives_find
from ellips_param import find_ellips_param
from etl import read_points_from_file, lists_to_excel
from find_params import find_param
from tsallis import Tsallian, ellips, pirsonian, simple_tsallis, simple_tsallis_

def find_dependece(q0):

    dhm = 0.5
    hm_list = np.arange(0.05, 6.0, dhm)
    iteration = 0

    # Создаем таблицу куда будем добавлять результат
    template = "hm, ampl_{}, {}t_{}, Gt_{}, msn_{}"

    columns = []
    if PIRSONIAN_MODE:
        columns_pirs = (template.format("pirs", "M", "pirs", "pirs", "pirs") + ", q0, App, dHpp").split(', ')
        columns += columns_pirs
    if TSALLIAN_MODE:
        columns_tsal = template.format("tsal", "q", "tsal", "tsal", "tsal").split(', ')
        columns += columns_tsal

    table_result = pd.DataFrame(columns=columns)
    tsal_intersection_result = pd.DataFrame(columns=['q0', 'hm', 'hm/G*'])

    while (iteration < ITERATION_DEPTH):

        for hm in hm_list:

            # Параметры для Тцаллиана, который будет искажаться
            params = {
                "Number of points": 10000, 
                "q": q0, 
                "G": 1.0, 
                "H_0": 3250.0, 
                "H_left": 3230.0, 
                "hm": hm, 
                "distortion": True
            }

            # Получаем объект тцаллиана фабрикой
            tsal = Tsallian().tsall_init(*list(params.values()))

            # Находим точку слева, чтобы "обрезать крылья" тцаллиана
            params["H_left"] = tsal.find_left()

            # Пересчитываем снова, с тем же количеством точек
            tsal_cropped = Tsallian().tsall_init(*list(params.values()))

            if TSALLIAN_MODE:
                params_tsal, msn_tsal = find_param(
                    hm,
                    simple_tsallis,
                    INITAL_AMPL_TSAL,
                    INITAL_Q_TSAL,
                    INITAL_G_TSAL,
                    tsal_cropped,
                    BOUNDS_TSAL
                )

                new_row = dict(
                     zip(columns_tsal, 
                         [hm, params_tsal[0], params_tsal[1], params_tsal[2], msn_tsal, q0, tsal_cropped.dHpp])
                )

                table_result = table_result._append(new_row, ignore_index=True)

                if params_tsal[1] <= 1.0:
                    hm_intersection = hm
                    print(f"Найдено пересечение с q=1 при q0={q0} на итерации {iteration} \n"
                        f"Найденные параметры: hm={hm}, hm/G*={hm/params_tsal[2]}")
                    if iteration == ITERATION_DEPTH - 1:
                        new_row = pd.Series([q0, hm, hm/params_tsal[2]], index=tsal_intersection_result.columns)
                        tsal_intersection_result  = tsal_intersection_result._append(new_row, ignore_index=True)
                        return tsal_intersection_result, table_result
                    break

            if PIRSONIAN_MODE:
                params_pirs, msn_pirs = find_param(
                    hm,
                    pirsonian,
                    INITAL_AMPL_PIRS,
                    INITAL_M_PIRS,
                    INITAL_G_PIRS,
                    tsal_cropped,
                    BOUNDS_PIRS
                )

                if params_pirs[1] >= 1000:
                    hm_intersection = hm
                    print(f"Найдено пересечение с M=1000 при q0={q0} на итерации {iteration} \n"
                          f"Найденные параметры: hm={hm}, hm/G*={hm/params_pirs[2]}"
                    )
                    params_ellips = find_ellips_param(
                        q0-1,
                        table_result['hm']/table_result['dHpp'],
                        1/table_result['Mt_pirs']
                    )

                    params_ellips, cov = curve_fit(
                        ellips, 
                        table_result['hm']/table_result['dHpp'], 
                        1/table_result['Mt_pirs'],
                        p0=[1, 1, 1, 1],
                        method='trf',
                        bounds=([0, 0, 0, 0], [2, 2, 2, 2])
                    )
                    return params_ellips, table_result

                new_row = dict(
                     zip(columns_pirs, 
                         [hm, params_pirs[0], params_pirs[1], params_pirs[2], 
                          msn_pirs, q0, tsal_cropped.App, tsal_cropped.dHpp])
                )

                table_result = table_result._append(new_row, ignore_index=True)

                writetype = "w" if np.where(hm_list == hm)[0] == 0 else "a"
            
                with open(f"params_pirsonian_q={q0}.out", f"{writetype}") as file:
                    file.write(f"{hm:.6e}\t{params_pirs[0]:.6e}\t{params_pirs[2]:.6e}\t{params_pirs[1]:.6e}\t"
                               f"{tsal_cropped.dHpp:.6e}\t{msn_pirs:.6e}\t{tsal_cropped.App:.6e}\t{q0:.6e}\n"
                    )
                


        hm_left = hm_intersection - 2*dhm
        hm_right = hm_intersection + 2*dhm
        dhm = dhm/10
        hm_list = np.arange(hm_left, hm_right, dhm)
        iteration += 1


def main():
    
    find_dependece(2.0)

    pool_size = 10

    # Создаем пул процессов
    with multiprocessing.Pool(pool_size) as pool:
        # Список задач (например, числа от 0 до 19)
        tasks = tuple(np.arange(1.05, 3, 0.05))

        # Распределение задач между процессами и сбор результатов
        results = pool.map(find_dependece, tasks)

        # Вывод результатов
        print(results)
    
    with open('results_dependence_pirs.pkl', 'wb') as f:
        pickle.dump(results, f)
        # plt.figure(figsize=(8, 5))
        # plt.scatter(hm_list, qt_list, label='Data')
        # plt.title('Curve Fitting Using Q-Gaussian Function')
        # plt.legend()
        # plt.savefig(f'plot(hm_list_qt_list_it={iteration}).png')
        # plt.close()

    # plt.figure(figsize=(8, 5))
    # plt.scatter(hm_list, qt_list, label='Data')
    # plt.plot(hm_list, first_derivative, label='Fitted Q-Gaussian', color='red')
    # plt.title('Curve Fitting Using Q-Gaussian Function')
    # plt.legend()
    # plt.show(block=True)
    # Plot the original data and the fitted curve
    # plt.figure(figsize=(8, 5))
    # plt.scatter(tsal_cropped.B, tsal_cropped.Y_norm, label='Data')
    # plt.plot(tsal_cropped.B, fitted_y, label='Fitted Q-Gaussian', color='red')
    # plt.title('Curve Fitting Using Q-Gaussian Function')
    # plt.legend()
    # plt.show(block=True)
    # plt.figure(figsize=(8, 5))
    # plt.scatter(hm_list, qt_list, label='Data')
    # plt.title('Curve Fitting Using Q-Gaussian Function')
    # plt.legend()
    # plt.show(block=True)

        # plt.figure(figsize=(8, 5))
        # plt.scatter(table_result['hm']/table_result['dHpp'], 1/table_result['Mt_pirs'], label='q = 2, G = 1')
        # plt.scatter(table_result['hm']/table_result['dHpp'], list(map(lambda x: ellips(x, params_ellips[0],params_ellips[1],params_ellips[2],params_ellips[3],params_ellips[4]), table_result['hm']/table_result['dHpp'])), label='ellips')
        # plt.text(0.1, 0.5, f"Параметры подгонки:\nP0: {params_ellips[0]}\nP1: {params_ellips[1]}\nP2: {params_ellips[2]}\nP3: {params_ellips[3]}\nP4: {params_ellips[4]}", ha='left', va='top')
        # plt.title('Зависимость 1/')
        # plt.legend()

        # plt.figure(figsize=(8, 5))
        # plt.scatter(table_result['hm']/table_result['dHpp'], 1/table_result['Mt_pirs'], label='q = 2, G = 1')
        # plt.scatter(table_result['hm']/table_result['dHpp'], list(map(lambda x: ellips(x, params_ellips[0],params_ellips[1],params_ellips[2],params_ellips[3]), table_result['hm']/table_result['dHpp'])), label='ellips')
        # plt.text(0.1, 0.5, f"Параметры подгонки:\nP2: {params_ellips[0]}\nP3: {params_ellips[1]}\nP4: {params_ellips[2]}\nP5: {params_ellips[3]}", ha='left', va='top')
        # plt.xlabel('hm/dHpp')
        # plt.ylabel('1/M')
        # plt.title('Зависимость 1/M от hm/dHpp для q=2, G=1')
        # plt.legend()

        # def quadratic_error(variables, *args):
        #     p2,p3,p4 = variables # p2 примерно равняетя 1/M = q-1 в относительных, p3 от 0 до 2, p4 от -0.5 до 0.5 
        #     x =args[0]
        #     y =args[1]
        #     y_pred = 0+p2*(1-((x-p4)/p3)**2)**0.5
        #     return np.sum((y - y_pred) ** 2)

        # initial_guess = [1, 1, 0.5]  # Начальное приближение для параметров a и b
        # bounds = bounds = [(0.001, 2), (0.01, 2), (0.01, 2)]
        # result = minimize(quadratic_error, initial_guess, method='L-BFGS-B', args=(df['col1'][:370]/df['col5'][:370], 1/df['col4'][:370]), bounds=bounds,  options={'maxfun': 500000, 'maxiter': 50000000, 'ftol': 1e-7, 'gtol': 1e-5})

        # import pandas as pd

        # # Загрузка данных из файла
        # df = pd.read_csv('param_mtc_s_p(s2a4)(q=3, G=1).out', sep='\t', names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'])

if __name__=="__main__":
    main()