import pandas as pd
import numpy as np
from BMS_Classifier import BMS_Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # Iris dataset
    X, y = load_iris(return_X_y=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, stratify=y)

    # Entrenamiento con un solo clasificador
    bmsC = BMS_Classifier(gamma=0.7)
    bmsC.fit(X_tr, y_tr)
    bmsC_pred = bmsC.predict(X_te)
    print(f"Accuracy Classifier: {round(accuracy_score(bmsC_pred, y_te), 2)}")

    # Entrenamiento con varios clasificadores(Bagging Ensemble)
    bmsE = BaggingClassifier(base_estimator=bmsC, n_estimators=10, n_jobs=-1)
    bmsE.fit(X_tr, y_tr)
    bmsE_pred = bmsE.predict(X_te)
    print(f"Accuracy Bagging: {round(accuracy_score(bmsE_pred, y_te), 2)}")


if __name__ == "__main__":
    main()
