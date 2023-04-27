import numpy as np

# Función para comprobar si la neurona se dispara
def firing_state(neuron, theta=1):
    return 1 if neuron >= theta else 0

if __name__ == '__main__':
    theta = 1 # Umbral
    gamma = 0.5 # Tasa de fuga
    i_ext = np.random.random() # Estimulo externo
    n = 10 # Número de neuronas
    v = np.zeros(n) # Neuronas
    spike_train = [] # Tren de pulsos

    print(f'Valor de i_ext: {i_ext}')

    for i in range(1, n):
        # Operación para obtener el valor de una neurona
        v[i] = gamma * v[i-1] * (1 - firing_state(v[i-1])) + i_ext

        # Comprobar si la neurona descanso
        if (v[i] > theta):
            spike_train.append(i)

    print(f'Values: {v}')
    print(f'Spike train: {spike_train}')
