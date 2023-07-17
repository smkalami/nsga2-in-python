import numpy as np
from copy import deepcopy
from itertools import chain

class NSGA2:
    """A class to implement the NSGA-II multi-objective optimization algorithm"""

    def __init__(self, max_iter = 100, pop_size = 100, p_crossover = 0.7, alpha = 0, p_mutation = 0.2, mu = 0.02, verbose = True):
        """Constructor for the NSGA-II object"""
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.alpha = alpha
        self.p_mutation = p_mutation
        self.mu = mu
        self.verbose = verbose

    def run(self, problem):
        """Runs the NSGA-II algorithm on a given problem."""
        
        # Extract Problem Info
        cost_function = problem['cost_function']
        n_var = problem['n_var']
        var_size = (n_var,) #np.array([1, n_var])
        var_min = problem['var_min']
        var_max = problem['var_max']

        # Number of offsprings/parents (multiple of 2)
        n_crossover = 2*int(self.p_crossover * self.pop_size / 2)

        # Number of Mutatnts
        n_mutation = int(self.p_mutation * self.pop_size)

        # Mutation Step Size
        sigma = 0.1 * (var_max - var_min)

        # Empty Individual
        empty_individual = {
            'position': None,
            'cost': None,
            'rank': None,
            'crowding_distance': None,
        }

        # Initialize Population
        pop = [deepcopy(empty_individual) for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            pop[i]['position'] = np.random.uniform(var_min, var_max, var_size)
            pop[i]['cost'] = cost_function(pop[i]['position'])

        # Non-dominated Sorting
        pop, F = self.non_dominated_sorting(pop)

        # Calculate Crowding Distance
        pop = self.calc_crowding_distance(pop, F)

        # Sort Population
        pop, F = self.sort_population(pop)

        # Main Loop
        for it in range(self.max_iter):
            
            # Crossover
            popc = [[deepcopy(empty_individual), deepcopy(empty_individual)] for _ in range(n_crossover//2)]
            for k in range(n_crossover//2):
                parents = np.random.choice(range(self.pop_size), size = 2, replace = False)
                p1 = pop[parents[0]]
                p2 = pop[parents[1]]
                popc[k][0]['position'], popc[k][1]['position'] = self.crossover(p1['position'], p2['position'])
                popc[k][0]['cost'] = cost_function(popc[k][0]['position'])
                popc[k][1]['cost'] = cost_function(popc[k][1]['position'])
                
            # Flatten Offsprings List
            popc = list(chain(*popc))
            
            # Mutation
            popm = [deepcopy(empty_individual) for _ in range(n_mutation)]
            for k in range(n_mutation):
                p = pop[np.random.randint(self.pop_size)]
                popm[k]['position'] = self.mutate(p['position'], sigma)
                popm[k]['cost'] = cost_function(popm[k]['position'])

            # Create Merged Population
            pop = pop + popc + popm

            # Non-dominated Sorting
            pop, F = self.non_dominated_sorting(pop)

            # Calculate Crowding Distance
            pop = self.calc_crowding_distance(pop, F)

            # Sort Population
            pop, F = self.sort_population(pop)

            # Truncate Extra Members
            pop, F = self.truncate_population(pop, F)

            # Show Iteration Information
            if self.verbose:
                print(f'Iteration {it + 1}: Number of Pareto Members = {len(F[0])}')

        # Pareto Front Population
        pareto_pop = [pop[i] for i in F[0]]
        
        return {
            'pop': pop,
            'F': F,
            'pareto_pop': pareto_pop,
        }
        
    def dominates(self, p, q):
        """Checks if p dominates q"""
        return all(p['cost'] <= q['cost']) and any(p['cost'] < q['cost'])

    def non_dominated_sorting(self, pop):
        """Perform Non-dominated Sorting on a Population"""
        pop_size = len(pop)

        # Initialize Domination Stats
        domination_set = [[] for _ in range(pop_size)]
        dominated_count = [0 for _ in range(pop_size)]

        # Initialize Pareto Fronts
        F = [[]]

        # Find the first Pareto Front
        for i in range(pop_size):
            for j in range(i+1, pop_size):
                # Check if i dominates j
                if self.dominates(pop[i], pop[j]):
                    domination_set[i].append(j)
                    dominated_count[j] += 1
                
                # Check if j dominates i
                elif self.dominates(pop[j], pop[i]):
                    domination_set[j].append(i)
                    dominated_count[i] += 1

            # If i is not dominated at all
            if dominated_count[i] == 0:
                pop[i]['rank'] = 0
                F[0].append(i)

        # Pareto Counter
        k = 0

        while True:
            
            # Initialize the next Pareto front
            Q = []
            
            # Find the members of the next Pareto front
            for i in F[k]:
                for j in domination_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        pop[j]['rank'] = k + 1
                        Q.append(j)
            
            # Check if the next Pareto front is empty
            if not Q:
                break
            
            # Append the next Pareto front
            F.append(Q)

            # Increment the Pareto counter
            k += 1

        return pop, F

    def calc_crowding_distance(self, pop, F):
        """Calculate the crowding distance for a given population"""

        # Number of Pareto fronts (ranks)
        parto_count = len(F)
        
        # Number of Objective Functions
        n_obj = len(pop[0]['cost'])

        # Iterate over Pareto fronts
        for k in range(parto_count):
            costs = np.array([pop[i]['cost'] for i in F[k]])
            n = len(F[k])
            d = np.zeros((n, n_obj))

            # Iterate over objectives
            for j in range(n_obj):
                idx = np.argsort(costs[:, j])
                d[idx[0], j] = np.inf
                d[idx[-1], j] = np.inf

                for i in range(1, n-1):
                    d[idx[i], j] = costs[idx[i+1], j] - costs[idx[i-1], j]
                    d[idx[i], j] /= costs[idx[-1], j] - costs[idx[0], j]

            # Calculate Crowding Distance
            for i in range(n):
                pop[F[k][i]]['crowding_distance'] = sum(d[i, :])

        return pop
    
    def sort_population(self, pop):
        """Sorts a population based on rank (in asceding order) and crowding distance (in descending order)"""
        pop = sorted(pop, key = lambda x: (x['rank'], -x['crowding_distance']))

        max_rank = pop[-1]['rank']
        F = []
        for r in range(max_rank + 1):
            F.append([i for i in range(len(pop)) if pop[i]['rank'] == r])

        return pop, F
    
    def truncate_population(self, pop, F, pop_size = None):
        """Truncates a population to a given size"""

        if pop_size is None:
            pop_size = self.pop_size

        if len(pop) <= pop_size:
            return pop, F

        # Truncate the population
        pop = pop[:pop_size]

        # Remove the extra members from the Pareto fronts
        for k in range(len(F)):
            F[k] = [i for i in F[k] if i < pop_size]

        return pop, F

    def crossover(self, x1, x2):
        """Performs crossover between two parents"""
        r_min = -self.alpha
        r_max = 1 + self.alpha
        r = np.random.uniform(r_min, r_max, x1.shape)
        y1 = r*x1 + (1-r)*x2
        y2 = r*x2 + (1-r)*x1
        return y1, y2
    
    def mutate(self, x, sigma):
        """Performs mutation on an individual"""
        n_var = x.size
        n_mu = np.ceil(self.mu*n_var)
        y = x.copy()
        
        J = np.random.choice(range(n_var), int(n_mu), replace=False)
        for j in J:
            y[j] += sigma*np.random.randn()
        
        return y
    
