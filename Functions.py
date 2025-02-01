from math import factorial as fac
import math


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
