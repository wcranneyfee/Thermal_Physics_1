from math import factorial as fac
import math
import scipy

k = scipy.constants.k
mu = scipy.constants.physical_constants['Bohr magneton'][0]

def binomial_dist(k, n):
    if n > k:
        raise ValueError('n cannot be greater than k')
    else:
        return fac(k)/(fac(n) * fac(k-n))


def einstein_solid_mult(N, q):
    return binomial_dist(N+q-1, q)


def stirling_approx(n):
    return ((n / math.e) ** n) * math.sqrt(2*math.pi*n)


def stirling_einstein_approx_mult(N, q, q_greater_than_N=False, N_greater_than_q=False, N_and_q_big=False):
    if q_greater_than_N is True:
        out = ((math.e * q) / N) ** N
    elif N_greater_than_q is True:
        out = 2
    elif N_and_q_big is True:
        out = 3
    else:
        raise ValueError('One boolean input must be true')
    return out


def einstein_interaction_probability(N_A, q_A, N_B, q_B):

    return (einstein_solid_mult(N_A, q_A) * einstein_solid_mult(N_B, q_B)) / einstein_solid_mult(N_A+N_B, q_A+q_B)


def paramagnetic_entropy(N, N_u, N_d):
    S_per_k = N*math.log(N) - N_u*math.log(N_u) - (N - N_u)*math.log(N-N_u)
    return S_per_k


def paramagnetic_temperature(x, a):
    T =  math.log(a * (1-(1/x)) / (1+(1/x)))
    return T

def magnetization(N, x):
    return N * mu * math.tanh(1/x)

def paramagnet_heat_capacity(x):
    C = (x**-2) / (math.cosh(1/x)**2)
    return C