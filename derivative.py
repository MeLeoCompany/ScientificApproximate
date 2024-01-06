from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative

def derivatives_find(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    # Создаем функцию для интерполяции данных
    f = interp1d(x, y, kind='linear')
    dx=1e-6
    # Функция для вычисления первой производной
    def first_derivative(x_point):
        return derivative(f, x_point, dx=1e-6)

    # Функция для вычисления второй производной
    def second_derivative(x_point):
        return derivative(f, x_point, dx=1e-6, n=2)

    # Вычисляем первую и вторую производные для набора точек
    x_range = x+dx
    first_derivatives = np.array([first_derivative(xp) for xp in x_range])
    second_derivatives = np.array([second_derivative(xp) for xp in x_range])

    # Находим минимум второй производной
    min_second_derivative = x_range[second_derivatives.argmin()]

    return first_derivatives, second_derivatives, min_second_derivative

