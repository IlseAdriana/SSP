import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize

class ProblemaEsfera(Problem):
    
    def __init__(self, n_var, xl, xu):
        super().__init__(n_var=n_var, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = np.sum(x**2, axis=1)


# Función que utiliza la metaheuristica que brindará el vector de pesos
def get_weights(n_var, pop_size, xl, xu, n_gen=10):
    problem = ProblemaEsfera(n_var=n_var, xl=xl, xu=xu)
    algorithm = PSO(pop_size=pop_size)

    solution = minimize(
        problem = problem,
        algorithm = algorithm,
        termination = ('n_gen', n_gen),
    )

    return (solution.X)