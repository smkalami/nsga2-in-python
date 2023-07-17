# NSGA-II in Python

This repository contains the implementation of NSGA-II algorithm in Python. The code is simple and easy to use. After defining the multi-objective optimization problem, it is passed to the NSGA-II object to be solved.

## About NSGA-II

The NSGA-II algorithm is a popular genetic algorithm for solving multi-objective optimization problems. It was first proposed by Deb et al. in 2002 as an improvement over the original Non-dominated Sorting Genetic Algorithm (NSGA).

NSGA-II works by iteratively creating a new population of solutions from the previous population. In each iteration, the following steps are performed:

1. The solutions in the previous population are sorted according to their non-domination rank.
2. The crowding distance of each solution is calculated.
3. The new population is created by selecting solutions from the previous population according to their non-domination rank and crowding distance.

The non-domination rank of a solution is a measure of how many other solutions it dominates. A solution that is dominated by no other solutions has a non-domination rank of 1. The crowding distance of a solution measures how close it is to other solutions in the same non-domination level.

The non-domination sorting procedure is used to rank the solutions in the population. This ranking is used to determine which solutions are selected for the new population. The crowding distance is used to control the diversity of the new population. Solutions with a high crowding distance are more likely to be selected for the new population, which helps to ensure that the new population is diverse. NSGA-II uses the rank and crowding distance, as primary and secindary selection criteria, respectively.

The new population is then used as the starting point for the next iteration of the algorithm. This process continues until a stopping criterion is met. The stopping criterion for the NSGA-II algorithm is typically a maximum number of iterations. However, other stopping criteria can also be used.

The pseudo-code of the NSGA-II algorithms follows.

```
// Initialization
P ← InitialPopulation() // Create Initial Population
NonDominatedSorting(P)  // Non-Dominated Ranking
CalcCrowdingDistance(P) // Calculate Crowding Distance

Repeat // Main Loop
  
    // Reproduction
    Parents ← SelectParents(P)        // Select Parents
    Q ← CrossoverAndMutation(Parents) // Get Offsprings
    R ← P + Q                         // Merging
    
    // Ranking and Selection
    NonDominatedSorting(R)  // Non-Dominated Ranking 
    CalcCrowdingDistance(R) // Calculate Crowding Distance
    P ← SelectSurvivors(R)  // Selecting Next Generation
    
    // Check for Termination
    If TerminationCriteriaSatisfied()
        Break // Exit Loop
  
Return P // Return Final Population
```

The solutions returned by the NSGA-II algorithm are not dominated by other solutions. This means that they are Pareto optimal, or at least sub-optimal, solutions to the problem being solved.

The decision maker can select any of the Pareto optimal solutions as the final solution to the problem. The choice of which solution to select will depend on the specific preferences of the decision maker.

The NSGA-II algorithm has several advantages over other multi-objective optimization algorithms. It is relatively fast and easy to implement. It is also not sensitive to the parameters of the algorithm.

The NSGA-II algorithm has been used to successfully solve a variety of multi-objective optimization problems. It is a powerful tool that can be used to find high-quality solutions to complex problems.

## Citing This Work

You can cite this code as follows:

**Mostapha Kalami Heris, NSGA-II in Python (URL: https://yarpiz.com), Yarpiz, 2023.**

