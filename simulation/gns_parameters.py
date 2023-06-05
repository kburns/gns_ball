

import numpy as np
import matplotlib.pyplot as plt


def scales_to_Gammas(k_center, k_width, growth_rate):
    kc = k_center
    kw = k_width
    s = growth_rate
    gamma2 = -2 * kc**2
    gamma0 = (kw**2 + gamma2)**2 / 4
    Gamma4 = 2 * s / gamma2 / (gamma0 - gamma2**2 / 4)
    Gamma0 = gamma0 * Gamma4
    Gamma2 = gamma2 * Gamma4
    return Gamma0, Gamma2, Gamma4

def Gammas_to_scales(Gamma0, Gamma2, Gamma4):
    gamma0 = Gamma0 / Gamma4
    gamma2 = Gamma2 / Gamma4
    kc = (gamma2 / -2)**0.5
    kw = (-gamma2 - 2 * gamma0**0.5)**0.5
    s = gamma2 * (gamma0 - gamma2**2/4) * Gamma4 / 2
    return kc, kw, s

def growth_rate(k, Gamma0, Gamma2, Gamma4):
    return -k**2 * (Gamma0 + Gamma2*k**2 + Gamma4*k**4)

def plot_growth(Gamma0, Gamma2, Gamma4):
    kc, kw, s = Gammas_to_scales(Gamma0, Gamma2, Gamma4)
    k = np.linspace(0, 2*kc, 1000)
    plt.figure()
    plt.axhline(0, c='k', lw=1, ls='dotted')
    plt.axhline(s, c='C1', lw=1, ls='dotted')
    plt.axvline(kc, c='C1', lw=1, ls='dotted')
    plt.axvline(kc+kw/2, c='C2', lw=1, ls='dotted')
    plt.axvline(kc-kw/2, c='C2', lw=1, ls='dotted')
    plt.plot(k, growth_rate(k, Gamma0, Gamma2, Gamma4), '-', c='C0')
    plt.yscale('symlog', linthresh=s)
    plt.xlabel('k')
    plt.ylabel('growth rate')
    plt.tight_layout()
    plt.savefig('growth.pdf')

if __name__ == "__main__":
    kc = 16
    kw = 2
    s = 1
    Gammas = scales_to_Gammas(kc, kw, s)
    print(Gammas)
    plot_growth(*Gammas)

