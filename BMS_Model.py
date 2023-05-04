import Problem
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize

# Función que contiene la metaheuristica que brindará el vector de pesos
def get_weights(n_var, pop_size, xl, xu, n_gen):
    problema = Problem.myProblem(n_var=n_var, xl=xl, xu=xu)
    algoritmo = PSO(pop_size=pop_size)

    resultado = minimize(
        problem = problema,
        algorithm = algoritmo,
        termination = ('n_gen', n_gen),
    )

    return (resultado.X)

# Función que contiene el modelo BMS
def bms_model(i_ext, sim_time = 100, theta = 1, gamma = 0.5):
    v = [] # Vector de ..
    spike_train = [] # Tren de pulsos

    v.append(0.) # Inicializamos el vector de ... con 0

    # Ciclo para obtener el tren de pulsos
    for k in range(1, sim_time):
        # Operación para obtener el valor de ...
        _v = gamma * v[-1] * (1 - (1 if v[-1] >= theta else 0)) + i_ext

        # Comprobar si la neurona descanso
        if (_v >= theta):
            spike_train.append(k)
            _v = theta

        v.append(_v)
    
    return spike_train


# Función para obtener la tasa de disparo correspondiende a cada clase
def get_firing_rates(n_class, weights, X, y):
    # Diccionario para almacenar la cantidad de pulsos de cada patron
    # correspondiente a una clase (tasa de disparo)
    firingRates = {i:[] for i in range(n_class)}

    # Ciclo para obtener las tasas de disparo por cada clase
    for _x, _y in zip(X, y):
        _iext = _x @ weights # Generar el estímulo a partir del patron actual y los pesos
        _fr = len(bms_model(_iext)) # Obtener la cantidad de disparos que hizo el patrón actual
        firingRates[_y].append(_fr) # Agregar la tasa de disparo del patrón a la clase correspondiente

    return firingRates


def main():
    X, y =  load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te =  train_test_split(X,y,test_size=0.1, stratify=y)

    dim = X.shape[1]
    n_class = len(np.unique(y))

    # Obtener el vector de pesos
    w = get_weights(n_var=dim, pop_size=15, xl=-10, xu=10, n_gen=10)

    # Obtener las tasas de disparo 
    firingRates = get_firing_rates(n_class, w, X_tr, y_tr)

    for c in range(n_class):
        print(f'Clase {c}, Media_fr: {np.mean(firingRates[c])}, Desv. Est.: {np.std(firingRates[c])}')
  

if __name__ == '__main__':
    main()