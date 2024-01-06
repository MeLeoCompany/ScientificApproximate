
import numpy as np
from scipy.optimize import curve_fit

from tsallis import pirsonian

def find_param(
    hm,
    func,
    initial_ampl, # параметры начальной точки
    initial_qM,
    inital_G,
    tsal_cropped,
    bounds # искаженный обрезанный тцаллиан
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