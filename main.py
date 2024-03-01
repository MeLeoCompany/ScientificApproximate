from functools import partial
import multiprocessing
import os
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from consts import BOUNDS_PIRS, BOUNDS_TSAL, INITAL_AMPL_PIRS, \
    INITAL_AMPL_TSAL, INITAL_G_PIRS, INITAL_G_TSAL, INITAL_M_PIRS, \
    INITAL_Q_TSAL, ITERATION_DEPTH, PIRSONIAN_MODE, TSALLIAN_MODE, \
    TSALLIAN_MODE_DAT, TWO_TSALLIAN_MANUAL_FUNMIN_CHECK, TWO_TSALLIAN_MANUAL_MODE

from derivative import derivatives_find
from ellips_param import find_ellips_param
from etl import read_points_from_file, lists_to_excel
from find_params import find_one_tsall_param, find_param, find_two_tsall_param
from tsallis import Tsallian, ellips, pirsonian, simple_tsallis, simple_tsallis_
from tsallis_fix_b import find_params_two_tsallis_fixB


def find_dependece(q0):

    dhm = 0.5
    hm_list = np.hstack((np.arange(0.05, 2.0, 0.05), np.arange(2.05, 5.0, 0.1)))
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
    if TWO_TSALLIAN_MANUAL_MODE:
        columns_tsal = "ampl_1, qt_1, Gt_1, Bres_1, msn_1, ampl_2, qt_2, Gt_2, Bres_2, msn_2, "
        columns += columns_tsal

    if TSALLIAN_MODE_DAT:
        # experimental_spectr = pd.read_csv(
        #     './bdpa/BDPA-3253-20-pw=1-rg=5-ma=0,1.dat', sep='\t',
        #     names=['B', 'Signal'])
        # find_one_tsall_param(
        #     experimental_spectr,
        # )
        table_parametrs = pd.DataFrame(
            columns=['hm', 'funmin']
        )
        folder_path = "./bdpa"
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        pattern = r'ma=([\d,]+)'
        file_names.remove('BDPA-3253-20-pw=1-rg=50-ma=0,05.dat')
        file_names.remove('BDPA-3253-20-pw=1-rg=5-ma=0,1.dat')
        for file in file_names:
            ma_str = re.search(pattern, file).group(1)
            ma = float(ma_str.replace(",", "."))
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as file:
                for i, line in enumerate(file):
                    if 'x-coordinate\tAmplitude' in line:
                        header_line = i
                        break
            experimental_spectr = pd.read_csv(
                file_path, skiprows=header_line+1,
                delimiter='\t', names=['B', 'Signal']
            )
            X_list = experimental_spectr['B']
            Y_list = experimental_spectr['Signal']/(
                max(experimental_spectr['Signal']) - min(experimental_spectr['Signal'])
            )

            # ma = ma/1.5
            # Ampl0_1 = 0.9997809556326396
            # C = 0.0009108415494761934

            # params_1 = {
            #     "q0": 2.1191387055143847,
            #     "G0": 0.5540674486297806,
            #     "B0": 3252.932789971564,
            #     "H_array": X_list,
            #     "hm": ma,
            # }
            # funmin = 4.2829304406946225e-06

            ma = ma/1.5
            Ampl0_1 = 0.990691551703254
            C = 0.00019618396712957776

            params_1 = {
                "q0": 2.131899242280023,
                "G0": 0.5620619750350033,
                "B0": 3252.91258671749,
                "H_array": X_list,
                "hm": ma,
            }
            # funmin = 1.0081684997897367e-05
            Y1 = Tsallian().tsall_init_new(*list(params_1.values()))
            Y_sum = Y1*Ampl0_1 + C
            funmin = np.sum((Y_sum-Y_list)**2)/len(Y_sum)

            new_row = pd.Series(
                [ma, funmin],
                index=table_parametrs.columns
            )

            table_parametrs = table_parametrs._append(new_row, ignore_index=True)

        table_parametrs.sort_values(by='hm')

    if TWO_TSALLIAN_MANUAL_FUNMIN_CHECK:

        table_parametrs = pd.DataFrame(
            columns=['hm', 'funmin']
        )

        folder_path = "./Q55"
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        file_names.remove('series-3250-100-pw=10-rg=5-ma=0,1.dat')
        pattern = r'ma=([\d,]+)'
        for file in file_names:
            ma_str = re.search(pattern, file).group(1)
            ma = float(ma_str.replace(",", "."))
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as file:
                for i, line in enumerate(file):
                    if 'x-coordinate\tAmplitude' in line:
                        header_line = i
                        break
            experimental_spectr = pd.read_csv(
                file_path, skiprows=header_line+1,
                delimiter='\t', names=['B', 'Signal']
            )
            X_list = experimental_spectr['B']
            Y_list = experimental_spectr['Signal']/(
                max(experimental_spectr['Signal']) - min(experimental_spectr['Signal'])
            )

            ma = ma/1.5
            Ampl0_1 = 1.0087859013402007
            Ampl0_2 = 0.18161408924683042
            C = -0.0036201733674702296

            params_1 = {
                "q0": 1.5232059438020693,
                "G0": 2.6044979730736957,
                "B0": 3255.419345520497,
                "H_array": X_list,
                "hm": ma,
            }

            params_2 = {
                "q0": 1.98927927011648,
                "G0": 1.9904610760699508,
                "B0": 3257.7496011670673,
                "H_array": X_list,
                "hm": ma,
            }
            Y1 = Tsallian().tsall_init_new(*list(params_1.values()))
            Y2 = Tsallian().tsall_init_new(*list(params_2.values()))
            Y_sum = Y1*Ampl0_1 + Y2*Ampl0_2 + C
            funmin = np.sum((Y_sum-Y_list)**2)/len(Y_sum)

            new_row = pd.Series(
                [ma, funmin],
                index=table_parametrs.columns
            )

            table_parametrs = table_parametrs._append(new_row, ignore_index=True)

        table_parametrs.sort_values(by='hm')

    if TWO_TSALLIAN_MANUAL_MODE:
        experimental_spectr = pd.read_csv(
            'series-3250-100-pw=10-rg=5-ma=0,1.dat', sep='\t',
            names=['B', 'Signal'])
        find_two_tsall_param(
            experimental_spectr,
        )
        table_parametr = pd.DataFrame(
            columns=['hm', 'G0', 'q0', 'B0', 'ampl0', 'G0_1', 'q0_1', 'B0_1', 'ampl0_1', 'c', 'funmin']
        )
        B_0 = 3255.536860862727
        B_1 = 3257.9813044900907
        folder_path = "./Q55"
        file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        file_names.remove('series-3250-100-pw=10-rg=5-ma=0,1.dat')
        pattern = r'ma=([\d,]+)'
        for file in file_names:
            ma_str = re.search(pattern, file).group(1)
            ma = float(ma_str.replace(",", "."))
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r') as file:
                for i, line in enumerate(file):
                    if 'x-coordinate\tAmplitude' in line:
                        header_line = i
                        break
            experimental_spectr = pd.read_csv(
                file_path, skiprows=header_line+1,
                delimiter='\t', names=['B', 'Signal']
            )
            X_list = experimental_spectr['B']
            Y_list = experimental_spectr['Signal']/(
                max(experimental_spectr['Signal']) - min(experimental_spectr['Signal'])
            )
            res = find_params_two_tsallis_fixB(
                X_list, Y_list, B_0, B_1
            )
            new_row = pd.Series(
                [ma, res.x[0], res.x[1], B_0, res.x[2],  res.x[3], res.x[4], B_1, res.x[5], res.x[6], res.fun],
                index=table_parametr.columns
            )
            table_parametr = table_parametr._append(new_row, ignore_index=True)

        table_parametr.sort_values(by='hm')

    table_result = pd.DataFrame(columns=columns)
    tsal_intersection_result = pd.DataFrame(columns=['q0', 'hm', 'hm/G*'])
    ellips_param_result = pd.DataFrame(columns=['q0', 'p2', 'p3', 'p4', 'mse'])

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
                         [hm, params_tsal[0], params_tsal[1], params_tsal[2],
                          msn_tsal, q0, tsal_cropped.dHpp])
                )

                table_result = table_result._append(new_row, ignore_index=True)

                if params_tsal[1] <= 1.0:
                    hm_intersection = hm
                    print(f"Найдено пересечение с q=1 при q0={q0} на итерации {iteration} \n"
                          f"Найденные параметры: hm={hm}, hm/G*={hm/params_tsal[2]}")
                    if iteration == ITERATION_DEPTH - 1:
                        new_row = pd.Series([q0, hm, hm/params_tsal[2]], index=tsal_intersection_result.columns)
                        tsal_intersection_result = tsal_intersection_result._append(new_row, ignore_index=True)
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
                          f"Найденные параметры: hm={hm}, hm/G*={hm/params_pirs[2]}")
                    params_ellips = find_ellips_param(
                        q0-1,
                        table_result['hm']/table_result['dHpp'],
                        1/table_result['Mt_pirs']
                    )

                    new_row = pd.Series(
                        [q0, params_ellips[0], params_ellips[1], params_ellips[2], params_ellips[3]],
                        index=ellips_param_result.columns
                    )
                    ellips_param_result = ellips_param_result._append(new_row, ignore_index=True)

                    # params_ellips, cov = curve_fit(
                    #     ellips,
                    #     table_result['hm']/table_result['dHpp'],
                    #     1/table_result['Mt_pirs'],
                    #     p0=[1, 1, 1, 1],
                    #     method='trf',
                    #     bounds=([0, 0, 0, 0], [2, 2, 2, 2])
                    # )
                    return ellips_param_result, table_result

                new_row = dict(
                     zip(columns_pirs,
                         [hm, params_pirs[0], params_pirs[1], params_pirs[2],
                          msn_pirs, q0, tsal_cropped.App, tsal_cropped.dHpp])
                )

                table_result = table_result._append(new_row, ignore_index=True)

                writetype = "w" if np.where(hm_list == hm)[0] == 0 else "a"

                with open(f"params_pirs_data(q)/params_pirsonian_q={q0:.3f}.out", f"{writetype}") as file:
                    file.write(
                        f"{hm:.6e}\t{params_pirs[0]:.6e}\t{params_pirs[2]:.6e}\t{params_pirs[1]:.6e}\t"
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

    with open('results_ellipses_pirs_distortion.pkl', 'wb') as f:
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


if __name__ == "__main__":
    main()
