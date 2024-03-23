from dataclasses import dataclass, field
from functools import partial
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
import torch

from consts import PEAK_TO_PEAK_RECURSION_DEPTH


def simple_tsallis(x, amplt, q, G):
    H_0 = 3250.0
    a1 = pow(2.0, q - 1.0) - 1.0
    a2 = -1.0 / (q - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - H_0) / G, 2), a2 - 1.0) * a1 * ((x - H_0) / np.power(G, 2))
    f_max = np.max(SS)
    St = amplt * SS / (2.0 * f_max)
    return St


def simple_tsallis_ampl(x, amplt, q, G):
    H_0 = 3250.0
    a1 = pow(2.0, q - 1.0) - 1.0
    a2 = -1.0 / (q - 1.0)
    SS = - np.power(1.0 + a1 * np.power((x - H_0) / G, 2), a2 - 1.0) * (x - H_0)
    St = amplt * SS
    return St


def simple_tsallis_(x, q, G, H_0, Yt, dY):
    a1 = pow(2.0, q - 1.0) - 1.0
    a2 = -1.0 / (q - 1.0)
    SS = 2.0 * a2 * np.power(1.0 + a1 * np.power((x - H_0) / G, 2.0), a2 - 1.0) * a1 * ((x - H_0) / np.power(G, 2))
    f_max = np.max(SS)
    St = Yt * (SS / (2.0 * f_max)) + dY
    return St


def simple_tsallis_torch(x, q, G, H_0, Yt, dY):
    if not (torch.tensor(1.0) <= q <= torch.tensor(3.0)) or \
       not (torch.tensor(0) <= G <= torch.tensor(4.0)) or \
       not (torch.tensor(3245) <= H_0 <= torch.tensor(3260)) or \
       not (torch.tensor(0) <= Yt <= torch.tensor(1)):
        return torch.tensor([float('inf')])
    a1 = torch.pow(2.0, q - 1.0) - 1.0
    a2 = -1.0 / (q - 1.0)
    SS = 2.0 * a2 * torch.pow(1.0 + a1 * torch.pow((x - H_0) / G, 2.0), a2 - 1.0) * a1 * ((x - H_0) / torch.pow(G, 2))
    f_max = torch.max(SS)
    St = Yt * (SS / (2.0 * f_max)) + dY
    return St


def pirsonian(x, amplt, M, G):
    H_0 = 3250.0
    a1 = pow(2, 1.0 / M) - 1.0
    SS = - ((x - H_0) / G**2) * np.power(1 + a1 * np.power((x - H_0)/G, 2), -M-1)
    f_max = np.max(SS)
    St = amplt * (SS / (2.0 * f_max))
    return St


def ellips(x, p2, p3, p4):
    return 0+p2*(1-((x-p4)**2)/p3)**0.5


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

    @classmethod
    def tsall_init_new(
        cls,
        q: float,
        G: float,
        H_0: float,
        H_array: np.ndarray,
        hm: float = None,
    ) -> 'Tsallian':
        tmp_1 = pow(2.0, q - 1.0) - 1.0
        beta = sp.beta(0.5, 1.0 / (q - 1.0) - 0.5)
        f_max = np.sqrt(tmp_1) / (G * beta)
        Hd_list = H_array - H_0
        Y = np.zeros(len(Hd_list))

        i = 0
        for Hd in Hd_list:
            integrand = partial(cls.integral_new, tmp1=tmp_1, Hd=Hd, hm=hm, G=G, q=q)
            result, _ = quad(integrand, -np.pi, np.pi)
            Y[i] = result * f_max
            i += 1

        App = np.max(Y) - np.min(Y)
        Y_norm = Y / App
        return Y_norm

    def distortion_spectr(self):
        Hd = -(self.H_0 - self.H_left)
        tmp_1 = pow(2.0, self.q - 1.0) - 1.0
        beta = sp.beta(0.5, 1.0 / (self.q - 1.0) - 0.5)
        f_max = np.sqrt(tmp_1) / (self.G * beta)
        self.Hd = np.linspace(Hd, self.H_0 - self.H_left, self.N)

        self.Y, Hd_left, Hd_right = self.find_sperctr(self.N, self.Hd, f_max, tmp_1)

        dHd = self.Hd[1] - self.Hd[0]
        Hd_list_left_peak = np.linspace(Hd_left-dHd, Hd_left+dHd, self.N//2)
        Hd_list_right_peak = np.linspace(Hd_right-dHd, Hd_right+dHd, self.N//2)
        Hd_list = np.hstack((Hd_list_left_peak, Hd_list_right_peak))

        self.find_spectr_param_recursion(
            self.N,
            Hd_list,
            f_max,
            tmp_1,
            PEAK_TO_PEAK_RECURSION_DEPTH
        )

        i = 0
        for Hd in self.Hd:
            integrand = partial(self.integral, tmp1=tmp_1, Hd=Hd)
            result, _ = quad(integrand, -np.pi, np.pi)
            self.Y[i] = result * f_max
            i += 1

        self.App = np.max(self.Y) - np.min(self.Y)
        self.dHpp = self.Hd[np.argmin(self.Y)] - self.Hd[np.argmax(self.Y)]
        self.Y_norm = self.Y / self.App
        self.B = self.Hd + self.H_0
        return self

    def find_sperctr(
        self: 'Tsallian',
        N: int,
        Hd_list: np.ndarray,
        f_max: float,
        tmp_1: float
    ):
        Y = np.zeros(N, dtype=float)
        i = 0
        for Hd in Hd_list:
            integrand = partial(self.integral, tmp1=tmp_1, Hd=Hd)
            result, _ = quad(integrand, -np.pi, np.pi)
            Y[i] = result * f_max
            i += 1
        return Y, Hd_list[np.argmax(Y)], Hd_list[np.argmin(Y)]

    def find_spectr_param_recursion(
        self: 'Tsallian',
        N: int,
        Hd_list: np.ndarray,
        f_max: float,
        tmp_1: float,
        recursion_depth_const: int
    ):
        Y, Hd_left, Hd_right = self.find_sperctr(N, Hd_list, f_max, tmp_1)
        if recursion_depth_const == 0:
            self.App = np.max(Y) - np.min(Y)
            self.dHpp = Hd_list[np.argmin(Y)] - Hd_list[np.argmax(Y)]
            return None
        dHd = Hd_list[1] - Hd_list[0]
        Hd_list_left_peak = np.linspace(Hd_left-dHd, Hd_left+dHd, N//2)
        Hd_list_right_peak = np.linspace(Hd_right-dHd, Hd_right+dHd, N//2)
        Hd_list = np.hstack((Hd_list_left_peak, Hd_list_right_peak))
        recursion_depth_const -= 1
        self.find_spectr_param_recursion(N, Hd_list, f_max, tmp_1, recursion_depth_const)

    def integral(
        self: 'Tsallian',
        x: float,
        tmp1: float,
        Hd: float
    ):
        tmp2 = (Hd + 0.5 * self.hm * np.sin(x)) / self.G
        res = np.sin(x) * pow(1.0 + tmp1 * tmp2 * tmp2, -1.0 / (self.q - 1.0))
        return res

    @staticmethod
    def integral_new(
        x: float,
        tmp1: float,
        Hd: float,
        hm: float,
        G: float,
        q: float
    ):
        tmp2 = (Hd + 0.5 * hm * np.sin(x)) / G
        res = np.sin(x) * pow(1.0 + tmp1 * tmp2 * tmp2, -1.0 / (q - 1.0))
        return res
