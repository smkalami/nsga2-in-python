from nsga2 import NSGA2
import moo_test_problems as mtp

# Problem Definition
problem = {
    'cost_function': mtp.MOP4,
    'n_var': 3,
    'var_min': -5,
    'var_max': 5,
}

# Initialize Algorithm
alg = NSGA2(
    max_iter = 50,
    pop_size = 100,
    p_crossover = 0.7,
    alpha = 0.1,
    p_mutation = 0.3,
    mu = 0.05,
    verbose = True,
)

# Solve the Problem
results = alg.run(problem)
pop = results['pop']
F = results['F']

# Plot Results
import numpy as np
import matplotlib.pyplot as plt
pf_costs = np.array([pop[i]['cost'] for i in F[0]])
plt.scatter(pf_costs[:,0], pf_costs[:,1])
plt.grid()
plt.xlabel('f1')
plt.ylabel('f2')
plt.show()
