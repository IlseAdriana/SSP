from sklearn.base import BaseEstimator
import numpy as np

class BMS(BaseEstimator):
    
    def __init__(self, weights, sim_time=100, theta=1, gamma=0.5):
        self.weights = weights
        self.sim_time = sim_time
        self.theta = theta
        self.gamma = gamma

    def fit(self, X, y):
        
        def generate_spikeTrain(i_ext):
            v = [] # Vector de ...
            spike_train = [] # Tren de pulsos

            v.append(0.) # Inicializamos el vector de ... con 0

            # Ciclo para obtener el tren de pulsos
            for k in range(1, self.sim_time):
                # Operación para obtener el valor de ...
                _v = self.gamma * v[-1] * (1 - (1 if v[-1] >= self.theta else 0)) + i_ext

                # Comprobar si la neurona descanso
                if (_v >= self.theta):
                    spike_train.append(k)
                    _v = self.theta

                v.append(_v)

            return spike_train
        
        n_class = len(np.unique(y)) # Número de clases
        
        # Diccionario para almacenar la cantidad de pulsos de cada patron
        # correspondiente a una clase (tasa de disparo)
        firing_rates = {i:[] for i in range(n_class)}

        # Ciclo para obtener las tasas de disparo por cada clase
        for _x, _y in zip(X, y):
            _iext = _x @ self.weights # Generar el estímulo a partir del patron actual y los pesos
            _fr = len(generate_spikeTrain(_iext)) # Obtener la cantidad de disparos que hizo el patrón actual
            firing_rates[_y].append(_fr) # Agregar la tasa de disparo del patrón a la clase correspondiente

        return firing_rates
    