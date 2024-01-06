import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from derivative import derivatives_find
from etl import read_points_from_file, lists_to_excel
from tsallis import Tsallian, pirsonian, simple_tsallis, simple_tsallis_

AUTO = True
ITERATION_NUM = 2

def main():
    dhm = 0.1
    hm_list = np.arange(0.1, 3.5, dhm)
    qt = 2.0
    Gt = 1.0
    amplt = 1.0
    iteration = 0
    results_izlom = []
    results_massive = []

    while (iteration < ITERATION_NUM):
        # H_0t = 3250.0
        qt_list = []
        Gt_list = []
        H_0t_list = []
        msn_list = []
        amplt_list = []

        for hm in hm_list:
            params = [10000, 2.0, 1.0, 3250.0, 3230.0, hm, True]

            tsal = Tsallian().tsall_init(*params)
            params[4] = tsal.find_left()
            tsal_cropped = Tsallian().tsall_init(*params)
            # initial_guess = [
            #     amplt+random.uniform(-amplt*0.05, amplt*0.05),
            #     qt+random.uniform(-qt*0.05, qt*0.05), 
            #     Gt+random.uniform(-Gt*0.05, Gt*0.05)]
            initial_guess = [1.0, 1.0, 1.0]

            if AUTO:
                # params, covariance = curve_fit(
                #     simple_tsallis, tsal_cropped.B, tsal_cropped.Y_norm, p0=initial_guess, method='trf',  bounds=([0.0001, 0.5, 0.08], [100, 3.0, 10.0])
                #     )
                params, covariance = curve_fit(
                    pirsonian, tsal_cropped.B, tsal_cropped.Y_norm, p0=initial_guess, method='trf',  bounds=([0.0001, 0.5, 0.08], [100, 1000000, 100])
                    )
            else:
                pass
                # initial_guess = [2.0, 1.0, 3253.0, 1.0, 0.0] 
                # B, Y = [], []
                # B, Y = read_points_from_file('spectrum (1).inp')
                # B, Y_norm = np.array(B), np.array(Y) / (2*np.max(Y))
                # params, covariance = curve_fit(simple_tsallis_, B, Y_norm, p0=initial_guess)
                # qt, Gt, H_0t, Yt, dY = params
                # fitted_y = simple_tsallis_(B, qt, Gt, H_0t, Yt, dY)
                # plt.figure(figsize=(8, 5))
                # plt.scatter(B, Y_norm, marker='o', edgecolors='b', facecolors='none', s=10)
                # plt.plot(B, fitted_y, label='Fitted Q-Gaussian', color='red')
                # plt.title('Curve Fitting Using Q-Gaussian Function')
                # plt.legend()
                # plt.show(block=True)

            amplt, qt, Gt = params
        
            fitted_y = pirsonian(tsal_cropped.B, amplt, qt, Gt)

            msn = np.mean((tsal_cropped.Y_norm - fitted_y) ** 2)

            qt_list.append(qt)
            Gt_list.append(Gt)
            amplt_list.append(amplt)
            msn_list.append(msn)

            print(f"hm={hm:.3f}, qt={qt:.4f}, Gt={Gt:.2f}, H_0t={amplt:.2f}, msn={msn:.8f}")


        
        # lists_to_excel(
        #     hm_list,
        #     qt_list,
        #     Gt_list,
        #     amplt_list,
        #     msn_list
        # )

        # first_derivative, second_derivative, min_second_derivative = derivatives_find(hm_list, qt_list)
        first_derivative = np.gradient(qt_list, hm_list)
        second_derivative = np.gradient(first_derivative, hm_list)
        argmax_second_der = np.argmax(second_derivative)
        hm_left = hm_list[argmax_second_der] - 2*dhm
        hm_right = hm_list[argmax_second_der] + 2*dhm

        iteration_result = {
            'iteration': iteration, 
            'hm_izlom': hm_list[argmax_second_der], 
            'q_izlom': qt_list[argmax_second_der],
            'G_izlom': Gt_list[argmax_second_der],
            'hm_izlom/G_izlom': hm_list[argmax_second_der]/Gt_list[argmax_second_der]
        }

        print(str(iteration_result))

        iteration += 1
        results_izlom.append(iteration_result)
        
        iteration_massive_result = {
            'iteration': iteration, 
            'hm_list': hm_list, 
            'qt_list': qt_list,
            'first_derivative': first_derivative,
            'second_derivative': second_derivative
        }
        
        results_massive.append(iteration_massive_result)
    
        plt.figure(figsize=(8, 5))
        plt.scatter(hm_list, qt_list, label='Data')
        plt.title('Curve Fitting Using Q-Gaussian Function')
        plt.legend()
        plt.savefig(f'plot(hm_list_qt_list_it={iteration}).png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(hm_list, first_derivative, label='Data')
        plt.title('Curve Fitting Using Q-Gaussian Function')
        plt.legend()
        plt.savefig(f'plot(hm_list_first_derivative_it={iteration}).png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(hm_list, second_derivative, label='Data')
        plt.title('Curve Fitting Using Q-Gaussian Function')
        plt.legend()
        plt.savefig(f'plot(hm_list_second_derivative_it={iteration}).png')
        plt.close()

        dhm = dhm/10
        hm_list = np.arange(hm_left, hm_right, dhm)

    with open('results_izlom.pkl', 'wb') as f:
        pickle.dump(results_izlom, f)

    with open('results_massive.pkl', 'wb') as f:
        pickle.dump(results_massive, f)

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