from dataclasses import dataclass, field
from functools import partial
import numpy as np
import scipy.special as sp
from scipy.integrate import quad

def simple_tsallis(x, amplt, q, G):
    H_0 = 3250.0
    a1 = pow(2.0, q - 1.0) - 1.0
    a2 = -1.0 / (q - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - H_0) / G, 2), a2 - 1.0) * a1 * ((x - H_0) / np.power(G, 2))
    f_max = np.max(SS)
    St = amplt * SS / (2.0 * f_max)
    return St

def simple_tsallis_(x, q, G, H_0, Yt, dY):
    a1 = pow(2.0, q - 1.0) - 1.0
    a2 = -1.0 / (q - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - H_0) / G, 2.0), a2 - 1.0) * a1 * ((x - H_0) / np.power(G, 2))
    f_max = np.max(SS)
    St = Yt * (SS / (2.0 * f_max)) + dY
    return St

def pirsonian(x, amplt, M, G):
    H_0 = 3250.0
    a1 = pow(2, 1.0 / M) - 1.0
    SS = - ((x- H_0) / G**2) * np.power(1 + a1* np.power((x - H_0)/G , 2), -M-1)
    f_max = np.max(SS)
    St = amplt * (SS / (2.0 * f_max))
    return St


@dataclass
class Tsallian:
    
    N: int = None
    q: float = None
    G: float = None
    H_0: float = 3250.0
    H_left: float = None
    hm: float = None
    Y: np.ndarray = field(init=False)
    Y_norm: np.ndarray = field(init=False)
    Hd: np.ndarray = field(init=False)
    B: np.ndarray = field(init=False)
    dHpp: float = None,
    App: float = None

    def __post_init__(self: 'Tsallian') -> 'Tsallian':
        self.Y = np.zeros(self.N, dtype=float)
        self.Hd = np.zeros(self.N, dtype=float)

    def find_left(self: 'Tsallian', coef: float = 0.0001) -> 'Tsallian':
        index = np.argmax(self.Y > np.max(self.Y)*coef)
        return self.H_0 + self.Hd[index]

    @staticmethod
    def tsall_init(
            N: int,
            q: float,
            G: float, 
            H_0: float,
            H_left: float,
            hm: float = None,
            distortion: bool = False
    ) -> 'Tsallian':
        tsallian = Tsallian(N, q, G, H_0, H_left, hm)
        if distortion:
            tsallian = tsallian.distortion_spectr()
        else:
            pass
        return tsallian

    def distortion_spectr(self):
        Hd = -(self.H_0 - self.H_left)
        tmp_1 = pow(2.0, self.q - 1.0) - 1.0
        beta = sp.beta(0.5, 1.0 / (self.q - 1.0) - 0.5)
        f_max = np.sqrt(tmp_1) / (self.G * beta)
        self.Hd = np.linspace(Hd, self.H_0 - self.H_left, self.N)
        i = 0
        for Hd in self.Hd:
            integrand = partial(self.integral, tmp1=tmp_1, Hd=Hd)
            result, _ = quad(integrand, -np.pi, np.pi)
            self.Y[i] = result * f_max
            i +=1

        self.App = np.max(self.Y) - np.min(self.Y)
        self.dHpp = self.Hd[np.argmin(self.Y)] - self.Hd[np.argmax(self.Y)]
        self.Y_norm = self.Y / self.App
        self.B = self.Hd + self.H_0
        return self

    def integral(
            self: 'Tsallian',
            x: float,
            tmp1: float,
            Hd: float
    ):
        tmp2 = (Hd + 0.5 * self.hm * np.sin(x)) / self.G
        res = np.sin(x) * pow(1.0 + tmp1 * tmp2 * tmp2, -1.0 / (self.q - 1.0))
        return res