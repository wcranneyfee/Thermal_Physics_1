import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

k = scipy.constants.k
e = scipy.constants.e
k_eV = k/e

def E_vib_energy(n):

    return ((1.03*n) - (0.03 * (n**2))) * 0.53 # eV

def get_z(E_list, T):

    z = 0
    for E in E_list:
        z = z + np.exp(-E/(k_eV*T))

    return z

""" Q3 """

plt.figure(1)

E_vib_arr = np.array([(E_vib_energy(n) / 0.53) for n in range(9)])
E_sho_arr = np.linspace(0, 8, 9)
x_1 = np.array([1 for x in range(9)])
x_2 = np.array([2 for a in range(9)])

plt.scatter(x_1, E_sho_arr, s=150, label="Simple Harmonic Oscillator")
plt.scatter(x_2, E_vib_arr, s=150, label="Anharmonic Oscillator", marker='s')
plt.ylabel(r'$E_{vib} / \epsilon$')
plt.xticks([])
plt.legend()
plt.savefig('Plots/Project 2/Q2')

plt.figure(2)
T_arr = np.linspace(0, 5100, 50)
T_arr[0] = 1

E_vib_arr = [E_vib_energy(n) for n in range(11)]

z_arr = []
for T in T_arr:
    z_arr.append(get_z(E_vib_arr, T))

plt.plot(T_arr, z_arr, label = r"$ln(z_{vib})$")
plt.xlabel("T [K]")
plt.ylabel(r"$z_{vib}$")
plt.savefig('Plots/Project 2/Z_vs_T')

plt.figure(3)
lnz_arr = [np.log(z) for z in z_arr]
plt.plot(T_arr, lnz_arr, label=r"$ln(z_{vib}$")
plt.xlabel("T [K]")
plt.ylabel(r"$ln(z_{vib})$")
plt.savefig('Plots/Project 2/lnz_vs_T')


plt.show()
