from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize


# Clase que representa el modelo BMS
class BMS:

    def __init__(self, sim_time = 1000, gamma=0.1, theta = 1):
      self.sim_time = sim_time      
      self.gamma = gamma
      self.theta = theta

    def simulate(self, i_ext):
      v = [] # Vector de voltaje de membrana
      spike_train = [] # Tren de pulsos

      v.append(0.) # Inicializamos el vector de voltaje con 0

      # Ciclo para obtener el tren de pulsos
      for k in range(1, self.sim_time):
          # Operación para obtener el valor del voltaje
          _v = self.gamma * v[-1] * (1 - (1 if v[-1] >= self.theta else 0)) + i_ext

          # Comprobar si la neurona descanso
          if (_v >= self.theta):
              spike_train.append(k)
              _v = self.theta # Ajustamos el voltaje al valor del umbral

          v.append(_v) # Agregamos el voltaje

      return spike_train, len(spike_train) / self.sim_time
    

# Clase que representa la metaheurística que proporciona el vector de pesos
class BMS_Training(Problem):
    
    def __init__(self, X_tr, y_tr, n_var, xl, xu):
        super().__init__(n_var=n_var, xl=xl, xu=xu)
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.neuron = BMS()
        self.n_class = len(np.unique(y_tr))

    def _evaluate(self, x, out, *args, **kwargs):
        fit = np.zeros(len(x))
        firing_rates = {i:[] for i in range(self.n_class)}

        for i, w in enumerate(x):
          # Ciclo para obtener las tasas de disparo por cada clase
          for _x, _y in zip(self.X_tr, self.y_tr):
              _iext = _x @ w # Generar el estímulo a partir del patron actual y los pesos
              _,_fr = self.neuron.simulate(_iext) # Obtener la cantidad de disparos que hizo el patrón actual
              firing_rates[_y].append(_fr) # Agregar la tasa de disparo del patrón a la clase correspondiente
          
          _m = np.zeros(self.n_class)
          _sd = np.zeros(self.n_class)

          for _k in firing_rates.keys():
            _m[_k] = np.mean(firing_rates[_k])
            _sd[_k] = np.std(firing_rates[_k])

          sum_dist = 0
          for j in range(len(_m)):
            for k in range(j + 1, len(_m)):
              tmp = _m[j] - _m[k]
              sum_dist += np.sqrt(np.dot(tmp, tmp))


          fit[i] = ((1 / sum_dist) if sum_dist > 0 else 0.0000001) + np.sum(_sd)

        out['F'] = fit
  

# Clase que representa el clasificador y nos da las predicciones
class BMS_Classifier(BaseEstimator):
    
    def __init__(self, X_tr, y_tr, sim_time=100, theta=1, gamma=0.5):
        self.neuron = BMS(sim_time, theta, gamma)
        self.X_tr = X_tr
        self.y_tr = y_tr
        self._m = None
        self.n_class = None

    def fit(self, X, y):
      problem = BMS_Training(self.X_tr, self.y_tr, n_var=X.shape[1], xl=np.repeat(-1,X.shape[1]), xu=np.repeat(1,X.shape[1]))
    
      algorithm = PSO(pop_size=30)

      solution = minimize(
          problem = problem,
          algorithm = algorithm,
          termination = ('n_gen', 10),
          verbose = True
      )
      self.n_class = len(np.unique(self.y_tr))
      self.w = solution.X

      firing_rates = {i:[] for i in range(len(np.unique(self.y_tr)))}
      for _x, _y in zip(self.X_tr, self.y_tr):
          _iext = _x @ self.w # Generar el estímulo a partir del patron actual y los pesos
          _,_fr = self.neuron.simulate(_iext) # Obtener la cantidad de disparos que hizo el patrón actual
          firing_rates[_y].append(_fr) # Agregar la tasa de disparo del patrón a la clase correspondiente
      self._m = np.zeros(len(np.unique(self.y_tr)))

      for _k in firing_rates.keys():
        self._m[_k] = np.mean(firing_rates[_k])      


    def predict(self,X):
      
      pred = np.zeros(X.shape[0])
      for j, x in enumerate(X):
        _iext = x @ self.w 
        _,_fr = self.neuron.simulate(_iext)
        dif = np.zeros(self.n_class)

        for i, m in enumerate(self._m):
          dif[i] = np.abs(_fr-m)
        pred[j] = np.argmin(dif)

        return pred
