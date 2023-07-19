import numpy as np
from pymoo.core.problem import Problem
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize


# Clase que representa la neurona basada en el modelo BMS
class BMS:
    # Función para generar los trenes de pulso y las tasas de disparo
    def __init__(self, sim_time=1000, gamma=0.5, theta=1):
        self.sim_time = sim_time
        self.gamma = gamma
        self.theta = theta

    def simulate(self, i_ext):
        v = []  # Vector de voltaje de membrana
        spike_train = []  # Tren de pulsos

        v.append(0.0)  # Inicializamos el vector de voltaje con 0

        # Ciclo para obtener el tren de pulsos
        for k in range(1, self.sim_time):
            # Operación para obtener el valor del voltaje
            _v = self.gamma * v[-1] * (1 - (1 if v[-1] >= self.theta else 0)) + i_ext

            # Comprobar si la neurona descanso
            if _v >= self.theta:
                spike_train.append(k)
                _v = self.theta  # Ajustamos el voltaje al valor del umbral

            v.append(_v)  # Agregamos el voltaje

        # Devolvemos el tren de pulsos y la tasa de disparo
        return spike_train, (len(spike_train) / self.sim_time)


# Clase que representa la metaheurística que proporciona el vector de pesos
class BMS_Training(Problem):
    def __init__(self, X, y, sim_time, gamma, theta, n_var, xl, xu):
        super().__init__(n_var=n_var, xl=xl, xu=xu)
        self.X = X
        self.y = y
        self.neuron = BMS(sim_time, gamma, theta)
        self.n_class = len(np.unique(y))

    def _evaluate(self, x, out, *args, **kwargs):
        fit = np.zeros(len(x))
        firing_rates = {i: [] for i in range(self.n_class)}

        for i, w in enumerate(x):
            # Ciclo para obtener las tasas de disparo por cada clase
            for _x, _y in zip(self.X, self.y):
                _iext = (
                    _x @ w
                )  # Generar el estímulo a partir del patron actual y los pesos
                _, _fr = self.neuron.simulate(
                    _iext
                )  # Obtener la cantidad de disparos que hizo el patrón actual
                firing_rates[_y].append(
                    _fr
                )  # Agregar la tasa de disparo del patrón a la clase correspondiente

            _m = np.zeros(self.n_class)
            _sd = np.zeros(self.n_class)

            # Ciclo para obtener la media y la desviacón estándar por cada clase
            for _k in firing_rates.keys():
                _m[_k] = np.mean(firing_rates[_k])
                _sd[_k] = np.std(firing_rates[_k])

            # Ciclo para obtener la distancia entre las tasas de disparo por clase
            sum_dist = 0
            for j in range(len(_m)):
                for k in range(j + 1, len(_m)):
                    # tmp = _m[j] - _m[k]
                    # sum_dist += np.sqrt(np.dot(tmp, tmp))
                    sum_dist += np.linalg.norm(_m[j] - _m[k])

            # Obtener la aptitud de cada clase
            fit[i] = ((1 / sum_dist) if sum_dist > 0 else 1000000) + (np.sum(_sd) if np.sum(_sd) > 0 else 1000000)

        out["F"] = fit
