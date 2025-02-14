import Functions
import matplotlib.pyplot as plt
import numpy as np
import scipy

mu = scipy.constants.physical_constants['Bohr magneton'][0]

plt.figure(1)
N = 10000
N_u = np.linspace(1, N-1, N-1)
N_d = N - N_u

S_arr = []
for i, N_u_val in enumerate(N_u):
    S_arr.append(Functions.paramagnetic_entropy(N, N_u_val, N_d[i]))

plt.plot(N_u, S_arr)
plt.xlabel(r'$U / \mu B$')
plt.ylabel('S/k')
plt.savefig('Plots/HW3a_fig1.png')

plt.figure(2)

T_arr = []
a = 1
N1 = np.linspace(-10, -a-0.001, 10000)
N2 = np.linspace(a+0.001, 10, 10000)
N = np.concatenate((N1, N2))

for N_val in N:
    T_arr.append(Functions.paramagnetic_temperature(x=N_val, a=a))

plt.scatter(N, T_arr)
plt.xlabel(r"$U / N \mu B$")
plt.ylabel(r"$kT / \mu B$")
plt.savefig('Plots/HW3a_fig2.png')

plt.figure(3)

M_arr = []
x = np.linspace(-10, 10, 1000)
for x_val in x:
    M_arr.append(Functions.magnetization(20, x_val))

plt.scatter(x, M_arr)
plt.xlabel(r"$kT / \mu B$")
plt.ylabel(r"$M / N \mu$")
plt.savefig('Plots/HW3a_fig3.png')

plt.figure(4)
C_arr = []

x = np.linspace(0,5,1000)
for x_val in x:
    C_arr.append(Functions.paramagnet_heat_capacity(x_val))

plt.plot(x, C_arr)
plt.xlabel('$kT / \mu B$')
plt.ylabel(r"$C / Nk$")
plt.savefig('Plots/HW3a_fig4.png')
plt.show()