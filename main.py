from functools import partial
import multiprocessing
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from derivative import derivatives_find
from etl import read_points_from_file, lists_to_excel
from find_params import find_param
from tsallis import Tsallian, pirsonian, simple_tsallis, simple_tsallis_

# Режим работы
AUTO = True
HM_HAND = 2.0

ITERATION_DEPTH = 2 # глубина поиска пересечения с нулём

# Начальные точки и будем ли аппроксимировать, а также bounds
PIRSONIAN_MODE = False
INITAL_AMPL_PIRS = 1.0
INITAL_M_PIRS = 1.0
INITAL_G_PIRS = 1.0
BOUNDS_PIRS = ([0.0001, 0.5, 0.08], [100, 3, 10])

TSALLIAN_MODE = True
INITAL_AMPL_TSAL = 1.0
INITAL_Q_TSAL = 2.0
INITAL_G_TSAL = 1.0
BOUNDS_TSAL = ([0.5, 0.5, 0.08], [10, 3.0, 100])

def find_dependece(q0):

    dhm = 0.1
    hm_list = np.arange(0.1, 6.0, dhm)
    iteration = 0

    # Создаем таблицу куда будем добавлять результат
    template = "hm, ampl_{}, {}t_{}, Gt_{}, msn_{}"

    columns = []
    if PIRSONIAN_MODE:
        columns_pirs = template.format("pirs", "M", "pirs", "pirs", "pirs").split(', ')
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
                     zip(columns_tsal, [hm, params_tsal[0], params_tsal[1], params_tsal[2], msn_tsal])
                )

                table_result = table_result._append(new_row, ignore_index=True)

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

                new_row = dict(
                     zip(columns_pirs, [hm, params_pirs[0], params_pirs[1], params_pirs[2], msn_pirs])
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
        
        hm_left = hm_intersection - 2*dhm
        hm_right = hm_intersection + 2*dhm
        dhm = dhm/10
        hm_list = np.arange(hm_left, hm_right, dhm)
        iteration += 1


def main():

    pool_size = 10

    # Создаем пул процессов
    with multiprocessing.Pool(pool_size) as pool:
        # Список задач (например, числа от 0 до 19)
        tasks = tuple(np.arange(1.05, 3, 0.05))

        # Распределение задач между процессами и сбор результатов
        results = pool.map(find_dependece, tasks)

        # Вывод результатов
        print(results)
    
    with open('results_dependence.pkl', 'wb') as f:
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

if __name__=="__main__":
    main()