# Режим работы
AUTO = True
HM_HAND = 2.0

ITERATION_DEPTH = 1  # глубина поиска пересечения с нулём

# Начальные точки и будем ли аппроксимировать, а также bounds
PIRSONIAN_MODE = False
INITAL_AMPL_PIRS = 1.0
INITAL_M_PIRS = 1.0
INITAL_G_PIRS = 1.0
BOUNDS_PIRS = ([0.0001, 0.5, 0.08], [100, 10000, 10])

TSALLIAN_MODE = False
INITAL_AMPL_TSAL = 1.0
INITAL_Q_TSAL = 2.0
INITAL_G_TSAL = 1.0
BOUNDS_TSAL = ([0.5, 0.5, 0.08], [10, 3.0, 100])

TWO_TSALLIAN_MANUAL_MODE = False

TWO_TSALLIAN_MANUAL_FUNMIN_CHECK = True

TSALLIAN_MODE_DAT = False

TSALLIAN_MODE_DAT_0 = False
# Глубина рекурсии для уточняющего поиска dHpp и App
PEAK_TO_PEAK_RECURSION_DEPTH = 1
