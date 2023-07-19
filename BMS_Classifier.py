import BMS_Model as bms_class
import numpy as np
from sklearn.base import BaseEstimator
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize


# Clase que representa el clasificador y nos da las predicciones
class BMS_Classifier(BaseEstimator):
    def __init__(self, sim_time=100, gamma=0.5, theta=1):
        self.sim_time = sim_time
        self.gamma = gamma
        self.theta = theta
        self.neuron = bms_class.BMS(self.sim_time, self.theta, self.gamma)
        self._m = None
        self.n_class = None

    def fit(self, X, y):
        problem = bms_class.BMS_Training(
            X,
            y,
            self.sim_time,
            self.gamma,
            self.theta,
            n_var=X.shape[1],
            xl=np.repeat(-1, X.shape[1]),
            xu=np.repeat(1, X.shape[1]),
        )

        algorithm = DE(
            sampling=LHS(),
            variant="DE/rand/1/bin",
            pop_size=30,
            CR=0.3,
            dither="vector",
            jitter=False,
        )

        solution = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", 10),
        )

        self.n_class = len(np.unique(y))
        self.w = solution.X

        firing_rates = {i: [] for i in range(len(np.unique(y)))}

        for _x, _y in zip(X, y):
            _iext = (
                _x @ self.w
            )  # Generar el estímulo a partir del patron actual y los pesos
            _, _fr = self.neuron.simulate(
                _iext
            )  # Obtener la cantidad de disparos que hizo el patrón actual
            firing_rates[_y].append(
                _fr
            )  # Agregar la tasa de disparo del patrón a la clase correspondiente
        self._m = np.zeros(len(np.unique(y)))

        for _k in firing_rates.keys():
            self._m[_k] = np.mean(firing_rates[_k])

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=int)
        for j, x in enumerate(X):
            _iext = x @ self.w
            _, _fr = self.neuron.simulate(_iext)
            dif = np.zeros(self.n_class)

            for i, m in enumerate(self._m):
                dif[i] = np.abs(_fr - m)

            pred[j] = np.argmin(dif)

        return pred
