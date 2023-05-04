import Problem
from BMS_Estimator import BMS
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    X, y =  load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te =  train_test_split(X,y,test_size=0.1, stratify=y)

    n_class = len(np.unique(y)) # Número de clases

    # Obtener el vector de pesos
    w = Problem.get_weights(n_var=X.shape[1], pop_size=10, xl=-10, xu=10)
    print(f'Weights: {w}')
    
    bms = BMS(w) # Instancia del estimador

    # Obtener las tasas de disparo del conjunto de entrenamiento
    fr_tr = bms.fit(X_tr, y_tr)

    print(f'Training set {"-"*25}')
    for c in range(n_class):
        print(f'Clase {c}, Media_fr: {np.mean(fr_tr[c])}, Desv. Est.: {np.std(fr_tr[c])}')

    # Obtener las tasas de disparo del conjunto de prueba
    fr_te = bms.fit(X_te, y_te)

    print(f'Test set {"-"*25}')
    for c in range(n_class):
        print(f'Clase {c}, Media_fr: {np.mean(fr_te[c])}, Desv. Est.: {np.std(fr_te[c])}')


if __name__ == '__main__':
    main()