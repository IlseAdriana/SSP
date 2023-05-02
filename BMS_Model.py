import Problem
import numpy as np
from sklearn.datasets import load_iris
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize

# Función que contiene la metaheuristica que brindará el vector de pesos para cada clase
def get_weights(n_var, pop_size, xl, xu, n_gen):
    problema = Problem.myProblem(n_var=n_var, xl=xl, xu=xu)
    algoritmo = PSO(pop_size=pop_size)

    resultado = minimize(
        problem = problema,
        algorithm = algoritmo,
        termination = ('n_gen', n_gen),
    )

    return (resultado.X)


def main():
    X, y = load_iris(return_X_y=True)
    n_classes = len(np.unique(y)) # Cantidad de clases

    patterns = [[] for _ in range(n_classes)] # Crear una lista por clase
    
    # Ciclo para guardar los patrones en la lista de su clase correspondiente
    for pattern, label in zip(X, y):
        patterns[label].append(pattern) 

    patterns = np.asarray(patterns, dtype=float) # Convertimos a un arreglo

    pop_size = 10 # Población del PSO
    xl = -10 # Límite inferior
    xu = 10 # Límite superior
    n_gen = 15 # Número de generaciones

    # Obtenemos los pesos (solución de la metaheurística)
    weights = np.asarray(get_weights(n_classes, pop_size, xl, xu, n_gen), dtype=float) 
    
    print(f'Proposed weights per class: {weights}')

    i_ext = [[] for _ in range(n_classes)]
    for i in range(n_classes):

        # Ciclo para obtener el producto-punto del patron con su peso correspondiente
        for pattern in patterns[i]:
            i_ext[i].append(pattern @ weights[i]) 

    i_ext = np.asarray(i_ext)

    i_ext_means = np.mean(i_ext, axis=1) # Medias de cada clase
    print(f'Means per class: \n{i_ext_means}')

    theta = 1 # Umbral
    gamma = 0.5 # Tasa de fuga
    spike_train = []
    v = np.zeros(n_classes, dtype=float) # Neuronas

    for i in range(1, n_classes):
        v[i] = gamma * v[i-1] * (1 - (1 if v[i-1] >= theta else 0)) + i_ext_means[i]

        if (v[i] > theta):
            spike_train.append(i)

    print(f'Spike train: {spike_train}')

if __name__ == '__main__':
    main()