import numpy as np
from pymoo.core.problem import Problem

class myProblem(Problem):
    
    def __init__(self, n_var, xl, xu):
        super().__init__(n_var=n_var, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = np.sum(x**2, axis=1)
