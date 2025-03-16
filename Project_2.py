import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

k = scipy.constants.k
e = scipy.constants.e
k_eV = k/e

T_arr = np.arange(1, 10000, 2)

def E_vib_energy(n):

    return ((1.03*n) - (0.03 * (n**2))) * 0.53 # eV

def get_z(E_list, T):
    """ Note that E must be given in eV here"""
    z = 0
    for E in E_list:
        z = z + np.exp(-E/(k_eV*T))

    return z

def get_U_arr(N, Z1_arr, T_arr):
    """Note that T is in kelvin and k is in eV, so U outputs in eV"""
    U_arr = []
    lnZ1_arr = np.log(Z1_arr)
    for i, (T, lnZ1), in enumerate(zip(T_arr, lnZ1_arr)):

        if i > 0 and i < len(T_arr) - 1:
            U = N * k_eV * (T**2) * ((lnZ1_arr[i+1] - lnZ1_arr[i-1]) / (T_arr[i+1] - T_arr[i-1]))
            U_arr.append(U)

    return U_arr

def get_Cv_arr(U_arr, T_arr):
    """ Must input temp in kelvin and thermal energy in electron volts"""
    Cv_arr = []  # eV/K
    for i, (T, U), in enumerate(zip(T_arr, U_arr)):

        if i > 0 and i < len(T_arr) - 1:
            C_v = (U_arr[i+1] - U_arr[i-1]) / (T_arr[i+1] - T_arr[i-1])
            Cv_arr.append(C_v)

    return Cv_arr

def get_Z_rot_arr(j_arr, T_arr):

    """The length of T_arr is the length of the output. j_arr determines what indices you are summing over within each
    individual Z_rot value"""
    Z_rot_arr = []
    for T in T_arr:
        z_rot = 0
        for j in j_arr:
            z_rot = z_rot + (2*j + 1) * np.exp((-j*0.0076*(j+1)) / (k_eV * T))

        Z_rot_arr.append(z_rot)

    return Z_rot_arr

plt.figure(1)

E_vib_arr = np.array([(E_vib_energy(n) / 0.53) for n in range(9)]) # eV per epsilon
E_sho_arr = np.linspace(0, 8, 9) # eV per epsilon
x_1 = np.array([1 for x in range(9)])
x_2 = np.array([2 for a in range(9)])

plt.scatter(x_1, E_sho_arr, s=150, label="Simple Harmonic Oscillator")
plt.scatter(x_2, E_vib_arr, s=150, label="Anharmonic Oscillator", marker='s')
plt.ylabel(r'$E_{vib} / \epsilon$')
plt.xticks([])
plt.legend()
plt.savefig('Plots/Project 2/Q2')

plt.figure(2)
# T_arr = np.linspace(0, 5100, 50) # K
T_arr[0] = 1 # K

E_vib_arr = [E_vib_energy(n) for n in range(11)]  # eV

z_arr = [get_z(E_vib_arr, T) for T in T_arr] # unitless

plt.plot(T_arr, z_arr, label = r"$ln(z_{vib})$")
plt.xlabel("T [K]")
plt.ylabel(r"$Z_{vib}$")
# plt.savefig('Plots/Project 2/Z_vib_vs_T')

plt.figure(3)
lnz_arr = [np.log(z) for z in z_arr] # unitless
plt.plot(T_arr, lnz_arr, label=r"$ln(z_{vib}$")
plt.xlabel("T [K]")
plt.ylabel(r"$ln(Z_{vib})$")
# plt.savefig('Plots/Project 2/lnz_vib_vs_T')

plt.figure(4)
N = 100
U_arr = get_U_arr(N, z_arr, T_arr) # eV
T_arr = T_arr[1:-1] # K
UperN_arr = [U/N for U in U_arr] # eV
plt.plot(T_arr, UperN_arr, label=r"$U/N$")
plt.xlabel("T [K]")
plt.ylabel(r"$U_{vib}/N$ [eV]")
# plt.savefig('Plots/Project 2/U_vib_perN_vs_T')

plt.figure(5)
Cv_arr = get_Cv_arr(U_arr, T_arr) # eV/K

Cv_vib_perNk = [Cv / (N * k_eV) for Cv in Cv_arr] # unitless
T_arr = T_arr[1:-1] # K
plt.plot(T_arr, Cv_vib_perNk, label=r"$C/Nk$")
plt.xlabel("T [K]")
plt.ylabel(r"$C_{vib}/Nk$")
# plt.savefig('Plots/Project 2/C_vib_perNk_vs_T')

# T_arr = np.concatenate((T_arr1, T_arr2))
#
plt.figure(6)
# # T_arr = np.linspace(0, 504, 253)  # K
# # T_arr[0] = 0.01  # K
#
# j_arr = np.linspace(0, 10, 11)
# z_rot_arr = get_Z_rot_arr(j_arr, T_arr) # unitless
# plt.plot(T_arr, z_rot_arr, label=r"$z_{rot}$")
# plt.xlabel("T [K]")
# plt.ylabel(r"$Z_{rot}$")
# # plt.savefig('Plots/Project 2/z_rot_vs_T_WRONG')
#
# plt.figure(7)
# U_arr = get_U_arr(N, z_rot_arr, T_arr)
# UrotPerN = [U/N for U in U_arr]
# T_arr = T_arr[1:-1]
# plt.plot(T_arr, UrotPerN, label=r"$U_{rot}/N$")
# plt.xlabel("T [K]")
# plt.ylabel(r"$U_{rot}/N$ [eV]")
# # plt.savefig('Plots/Project 2/U_rot_PerN_vs_T_WRONG')
#
# plt.figure(8)
# Cv_arr = get_Cv_arr(U_arr, T_arr)
# CvperNk = [Cv / (N * k_eV) for Cv in Cv_arr]
# T_arr = T_arr[1:-1]
# plt.plot(T_arr, CvperNk, label=r"$C/Nk$")
# plt.xlabel("T [K]")
# plt.ylabel(r"$C_{rot}/Nk$")
# # plt.savefig('Plots/Project 2/C_rot_perNk_vs_T_WRONG')
#
# plt.figure(9)

T_arr = np.arange(1, 10000, 2)
# T_arr = np.linspace(0, 504, 253)
# T_arr = T_arr[8:]

j_ortho = np.linspace(1, 101, 51) # only odd indices
j_para = np.linspace(0, 100, 51) # only even indices

z_ortho_arr = get_Z_rot_arr(j_ortho, T_arr)
z_para_arr = get_Z_rot_arr(j_para, T_arr)

plt.plot(T_arr, z_ortho_arr, label=r"$Z_{ortho}$")
plt.plot(T_arr, z_para_arr, label=r"$Z_{para}$")
plt.xlabel("T [K]")
plt.ylabel(r"$Z_{rot}$")
plt.legend()
plt.savefig('Plots/Project 2/Z_ortho_and_Z_para_vs_T')

plt.figure(10)
U_ortho = get_U_arr(N, z_ortho_arr, T_arr)
U_ortho_perN = np.array([U/N for U in U_ortho])

U_para = get_U_arr(N, z_para_arr, T_arr)
U_para_perN = np.array([U/N for U in U_para])

U_net_perN = U_ortho_perN + U_para_perN

T_arr = T_arr[1:-1]
plt.plot(T_arr, U_ortho_perN, label=r"$U_{ortho}$")
plt.plot(T_arr, U_para_perN, label=r"$U_{para}$")
plt.plot(T_arr, U_net_perN, label=r"$U_{net}$")
plt.xlabel("T [K]")
plt.ylabel(r"$U_{rot}/N$ [eV]")
plt.legend()
# plt.savefig('Plots/Project 2/U_ortho_and_U_para_vs_T')

plt.figure(11)
Cv_ortho = get_Cv_arr(U_ortho, T_arr)
Cv_ortho_perNk = np.array([Cv / (N * k_eV) for Cv in Cv_ortho])

Cv_para = get_Cv_arr(U_para, T_arr)
Cv_para_perNk = np.array([Cv / (N * k_eV) for Cv in Cv_para])

for n, C in enumerate(Cv_ortho_perNk):
    if C < 0:
        Cv_ortho_perNk[n] = 0

Cv_net_perNk = 0.75*Cv_ortho_perNk + 0.25*Cv_para_perNk

T_arr = T_arr[1:-1]
plt.plot(T_arr, Cv_ortho_perNk, label=r"$C_{ortho}/Nk$")
plt.plot(T_arr, Cv_para_perNk, label=r"$C_{para}/Nk$")
plt.plot(T_arr, Cv_net_perNk, label=r"$C_{net}/Nk$")

plt.xlabel("T [K]")
plt.ylabel(r"$C_{rot}/Nk$")
plt.legend()
# plt.savefig('Plots/Project 2/Cv_ortho_and_Cv_para_vs_T')

plt.figure(12)

c_v_net = 3/2 + Cv_net_perNk + Cv_vib_perNk

plt.plot(T_arr, c_v_net, label=r"$C_{vib}/Nk$")
plt.xscale('log')
plt.xlabel('T [K]')
plt.ylabel(r"$C_v/Nk$")
plt.savefig('Plots/Project 2/C_net_log')
plt.show()




