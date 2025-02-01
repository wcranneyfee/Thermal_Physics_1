import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

g = sp.constants.g
k = sp.constants.k
R = sp.constants.physical_constants['molar gas constant'][0]

""" Problem A """


def adiabatic_expansion(V_arr: np.ndarray, P_i: [float, int], V_i: [float, int], f: int):
    gamma = (f + 2) / 2

    P_arr = (P_i * (V_i ** gamma)) / (V_arr ** gamma)

    return P_arr


def isothermal_expansion(V_arr: np.ndarray, P_i: [float, int], V_i: [float, int]):
    P_arr = (P_i * V_i) / V_arr

    return P_arr


def ideal_gas_find_T(P, V, n):
    T = (P*V) / (n*R)
    return T


V_in = np.linspace(1, 3, 100)

P_adia_3f = adiabatic_expansion(V_in, 1, 1, 3)
P_adia_5f = adiabatic_expansion(V_in, 1, 1, 5)
P_adia_7f = adiabatic_expansion(V_in, 1, 1, 7)
P_isoth = isothermal_expansion(V_in, 1, 1)

plt.figure(1)
plt.plot(V_in, P_adia_3f,  c='orange', label='adiabatic f=3')
plt.plot(V_in, P_adia_5f,  c='orange', label='adiabatic f=5', linestyle='-.')
plt.plot(V_in, P_adia_7f, c='orange', label='adiabatic f=7', linestyle='--')
plt.plot(V_in, P_isoth, c='purple', label='isothermal')
plt.vlines(1, 0, 1, colors='red', label='isobaric')
plt.hlines(1, 1, 3, colors='blue', label='isochoric')

plt.legend()
plt.xlabel(r'Volume [$m^3]$')
plt.ylabel('Pressure [atm]')
plt.savefig('Plots/ProblemA')

"""Since isothermal lines do not change as degrees of freedom change, there is only one isothermal line. However, there
are three adiabatic lines. As f increases, The adiabtic line becomes steeper. This is because as a gas is adiabatically
 compressed, it has more ways to store energy, and so the temperature increases at a slower rate when f is higher. The
 isochoric temperature increase could be caused by heating a gas in a sealed container with fixed volume. An isobaric
 could be caused by a gas being heated and pushing on a piston. The volume increases to keep the pressure constant. An
adiabatic expansion might be caused by a quickly expanding gas pushing on a piston. No heat is exchanged, so the 
pressure must decrease as volume increases. An isothermal expansion could be caused ice melting. As ice melts, 
its temperature remains the same."""



"""Problem B"""


def atmospheric_pressure(Z_in, P_0, m, T):
    P_out = P_0 * np.exp((-m*g*Z_in) / (R*T))
    return P_out


def atmospheric_temperature(Z_in, P_0, V_0, T_0, f):
    gamma = (f + 2) / 2
    P_in = atmospheric_pressure(Z_in, P_0, 0.028, 300)

    V_arr = ((P_0 * (V_0**gamma)) / P_in)
    T = (V_0 * (T_0 ** (f/2)) / V_arr) ** (2/f)

    return T


Z_arr = np.linspace(0, 9500, 100)  # m

pressure = atmospheric_pressure(Z_arr, 1, 0.028, 280)

plt.figure(2)
plt.plot(Z_arr, pressure, label='atmospheric pressure')
plt.xlabel('Elevation [m]')
plt.ylabel('Pressure [atm]')
plt.legend()
plt.savefig('Plots/ProblemB1')

plt.figure(3)
temp = atmospheric_temperature(Z_arr, 101325, 1, 300, 5)
plt.plot(Z_arr, temp)
plt.xlabel('Elevation [m]')
plt.ylabel('Temperature [K]')
plt.savefig('Plots/ProblemB_part2')

"""As the gas rises, the temperature decreases almost linearly. As the gas rises, pressure decreases, and so the gas' 
volume increases. This means the gas is doing work on the environment, which means it is giving away energy. This means 
that delta U is negative, which means that the change in temperature is negative. There is much more rain on the western
 side of Himalayas because it is the windward side. So rain is blown up the western side of the Himalayas, but it then
freezes and is stuck there, and so no rain gets to the eastern, or leeward side of the mountain."""



"""Problem C """


def pressure_of_V_and_T(V_in, T_in, n):
    return (n*R*T_in) / V_in


fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

vol = np.arange(.1, 2, 0.1)  # m^3
temp = np.arange(200, 600, 20)  # K
vol, temp = np.meshgrid(vol, temp, indexing='ij')

Z = pressure_of_V_and_T(vol, temp, 1)  # Pa
ax.plot_surface(vol, temp, Z)
ax.set_xlabel(r'Volume [$m^3$]')
ax.set_ylabel('Temperature [K]')
ax.set_zlabel('Pressure [Pa]')
plt.savefig('Plots/ProblemC_3D')

plt.figure(5)
temp = np.linspace(200, 600, 1000)
pressure_1 = pressure_of_V_and_T(0.2, temp, 1)
pressure_2 = pressure_of_V_and_T(1, temp, 1)

plt.plot(temp, pressure_1, label=r'V = 0.2 $[m^3]$')
plt.plot(temp, pressure_2, label=r'V = 1 $[m^3]$')

plt.xlabel('Temperature [K]')
plt.ylabel('Pressure [Pa]')
plt.legend()
plt.savefig('Plots/ProblemC_part2')

plt.figure(6)
vol = np.linspace(0.1, 2, 1000)
pressure_1 = pressure_of_V_and_T(vol, 300, 1)
pressure_2 = pressure_of_V_and_T(vol, 500, 1)

plt.plot(vol, pressure_1, label='T = 300 [K]')
plt.plot(vol, pressure_2, label='T = 500 [K]')

plt.xlabel(r'Volume [$m^3$]')
plt.ylabel('Pressure [Pa]')
plt.legend()
plt.savefig('Plots/ProblemC_part3')

"""You can see that each 2D plot is a cross section of the 3D surface. 
(𝝏𝑷/𝝏𝑻)𝑽 is the cross section with constant volume, while (𝝏𝑷/𝝏𝑽)𝑻 is the cross section with constant temperature. This 
reveals what a partial derivative actually is: holding one variable constant while varying another."""

"""Problem D"""

D = 1
del_x = 1.5
del_t = 0.1


def N_step(N_i1, N_i2, N_i3):
    N = N_i2 - (((D*del_t) / (del_x**2)) * (2*N_i2 - N_i1 - N_i3))
    return N


def N_2d_arr(x_arr, t_arr, N):

    for r, t_val in enumerate(t_arr):
        for c, x_val in enumerate(x_arr):

            if c == 0:
                N[r][0] = 0

            elif c == len(x_arr)-1 or r == len(t_arr)-1:
                continue

            else:
                N[r+1][c] = N_step(N[r][c-1], N[r][c], N[r][c+1])

    return N


x = np.arange(0, del_x*20, del_x)
print(len(x))
t = np.arange(0, del_t*100, del_t)
print(len(t))

N_ini = np.ones(shape=(len(t), len(x)))
N_ini[0][0:int(len(x) / 2)] = 0

N = N_2d_arr(x, t, N_ini)
x_mesh, t_mesh = np.meshgrid(x, t)

fig = plt.figure(7)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(x_mesh, t_mesh, N)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('N')
plt.savefig('Plots/ProblemD')
plt.show()

fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')

N_ini = np.ones(shape=(len(t), len(x)))*.75
N_ini[0][7:12] = 0.2

ax.plot_surface(x_mesh, t_mesh, N_ini)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('N')
plt.savefig('Plots/ProblemD_part_2')

"""Increasing D, delta x, or delta t all make the """
