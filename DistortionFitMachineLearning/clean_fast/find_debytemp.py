import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import math
import xrayutilities as xu


crystal = xu.materials.Crystal.fromCIF('/home/mariolb/repos/CrystalClearFit/DistortionFit/cif/P63_giant_cell.cif')
m = 0.0
im = 0
for a, _, _, b in crystal.lattice.base():
    m += a.weight
    im += 1
m = m / float(im)

print(f"m: {m}")


def Debye1(x):
    """
    function to calculate the first Debye function [1]_ as needed
    for the calculation of the thermal Debye-Waller-factor
    by numerical integration

    .. math:: D_1(x) = (1/x) \int_0^x t/(\exp(t)-1) dt

    Parameters
    ----------
    x : float
        argument of the Debye function

    Returns
    -------
    float
        D1(x)  float value of the Debye function

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Debye_function
     """

    def __int_kernel(t):
        """
        integration kernel for the numeric integration
        """
        y = t / (np.exp(t) - 1)
        return y

    if x > 0.:
        integral = scipy.integrate.quad(__int_kernel, 0, x)
        d1 = (1 / float(x)) * integral[0]
    else:
        integral = (0, 0)
        d1 = 1.

    return d1


temp = 673  # Kelvin
# W(q) = 3/2* hbar^2*q^2/(m*kB*tD) * (D1(tD/T)/(tD/T) + 1/4)
# DWF = exp(-W(q)) consistent with Vaclav H. and several books
hbar = scipy.constants.hbar
kb = scipy.constants.Boltzmann
angs = 1.0e-10
cell_length_a = 7.32000
cell_length_b = 11.04501
cell_length_c = 5.52250
volume = cell_length_a * cell_length_b * cell_length_c
volume = volume * (angs ** 3)  # m^3
n_atoms = 32
sound_v = 6000  # m/s
tD = (hbar * sound_v)/kb * ((6*n_atoms*np.pi**2)/volume) ** (1 / 3)  # Debye temperature
print(f"tD: {tD}")

d_1 = Debye1(tD / temp)
print(f"d_1: {d_1}")
x = tD / temp
exp_fact = 3 / 2. * hbar ** 2 * 1.0e20 / (m * kb * tD) * (Debye1(x) / x + 0.25)
print(f"exp_fact: {exp_fact}")
example_qs = np.linspace(0.1, 10, 100)
dwf = np.exp(-exp_fact * example_qs ** 2)
plt.plot(example_qs, dwf)
plt.xlabel('q (1/A)')
plt.ylabel('DWF')
plt.title('Debye-Waller factor')
plt.grid()
plt.savefig('debye_waller.png')
