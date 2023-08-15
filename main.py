import pandas as pd
import numpy as np
from BMS_Classifier import BMS_Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import load_datasets as datasets


def main():
    datasets_names = []

    with open("datasets_loading.txt", mode="r") as file:
        for line in file:
            line = line.replace("\n", "")
            datasets_names.append(line)

    file.close()

    # Entrenador Base
    bmsC = BMS_Classifier(gamma=0.5)

    # Clasificador Bagging
    bmsE = BaggingClassifier(base_estimator=bmsC, n_estimators=10, n_jobs=-1)

    # Diccionario para guardar los valores obtenidos por cada dataset
    results = {i: {} for i in range(len(datasets_names))}

    # Ciclo para probar todos los datasets
    for i, a_dataset in enumerate(datasets_names):
        X, y = eval(a_dataset)

        # Separar en conjunto de entrenamiento y de prueba
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, stratify=y)

        # Entrenamiento con un solo clasificador
        bmsC.fit(X_tr, y_tr)
        bmsC_pred = bmsC.predict(X_te)
        acc_classifier = round(accuracy_score(bmsC_pred, y_te), 2)
        results[i] = {"Classifier": acc_classifier}

        # Entrenamiento con varios clasificadores(Bagging Ensemble)
        bmsE.fit(X_tr, y_tr)
        bmsE_pred = bmsE.predict(X_te)
        acc_bagging = round(accuracy_score(bmsE_pred, y_te), 2)
        results[i].update({"Bagging": acc_bagging})

    for key, values in results.items():
        print(f"Dataset #{key+1} | Accuracies: {values}")


if __name__ == "__main__":
    main()
