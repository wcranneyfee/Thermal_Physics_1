from Functions import einstein_interaction_probability
import numpy as np
import matplotlib.pyplot as plt

q_tot = 50
x_arr = np.linspace(1, q_tot, q_tot, dtype=int)
y_arr = [einstein_interaction_probability(20, q_A, 80, q_tot-q_A) for q_A in x_arr]

plt.figure(1)
plt.scatter(x_arr, y_arr)
plt.xlabel(r'$q_A$')
plt.ylabel('Probability')
plt.savefig('Plots/HW2-10A')
