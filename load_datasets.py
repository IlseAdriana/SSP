import numpy as np
import pandas as pd


def load_wdbc():
    df = pd.read_csv("datasets/breast-cancer-wisconsin/wdbc.data", sep=",", header=None)
    X = np.asarray(df.iloc[:, 2:], dtype=float)
    y = np.where(df.iloc[:, 1] == "B", 0, 1)

    return X, y


def load_wobc():
    df = pd.read_csv("datasets/breast-cancer-wisconsin/wobc.data", sep=",", header=None)
    X = np.asarray(df.iloc[:, 1:-1], dtype=int)
    y = np.where(df.iloc[:, -1] == 2, 0, 1)

    return X, y


def load_wpbc():
    df = pd.read_csv("datasets/breast-cancer-wisconsin/wpbc.data", sep=",", header=None)
    X = np.asarray(df.iloc[:, 3:], dtype=float)
    y = np.where(df.iloc[:, 1] == "N", 0, 1)

    return X, y


if __name__ == "__main__":
    X, y = load_wdbc()
    print(X)
    print(y)

    print("*" * 40)
    X, y = load_wobc()
    print(X)
    print(y)

    print("*" * 40)
    X, y = load_wpbc()
    print(X)
    print(y)
