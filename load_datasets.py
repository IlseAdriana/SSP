import numpy as np
import pandas as pd


def load_wdbc():
    df = pd.read_csv("datasets/breast-cancer-wisconsin/wdbc.data", sep=",", header=None)
    X = np.asarray(df.iloc[:, 2:])
    y = np.where(df.iloc[:, 1] == "B", 0, 1)

    return X, y


def load_fertility():
    df = pd.read_csv("datasets/fertility/fertility.txt", sep=",", header=None)
    ncols = df.shape[1] - 1
    X = np.asarray(df.iloc[:, :ncols], dtype=float)
    y = np.where(df.iloc[:, ncols] == "N", 0, 1)

    return X, y


def load_hbsc():
    df = pd.read_csv("datasets/haberman/haberman.data", sep=",", header=None)
    ncols = df.shape[1] - 1
    X = np.asarray(df.iloc[:, :ncols], dtype=int)
    y = np.where(df.iloc[:, ncols] == 2, 0, 1)

    return X, y


def load_vr():
    df = pd.read_csv("datasets/lsvt-voice-rehabilitation/vr.csv", sep=",")
    ncols = df.shape[1] - 1
    X = np.asarray(df.iloc[:, :ncols], dtype=float)
    y = np.where(df.iloc[:, ncols] == 2, 1, 0)

    return X, y


def load_parkinson2():
    df = pd.read_csv("datasets/parkinson/parkinson2.data", sep=",")
    ncols = df.shape[1] - 1
    class_column = df.pop("status")
    df.insert(loc=ncols, column=class_column.name, value=class_column.values)
    X = np.asarray(df.iloc[:, 1:ncols], dtype=float)
    y = np.asarray(df.iloc[:, ncols])

    return X, y


def load_parkinson1():
    df = pd.read_csv(
        "datasets/parkinson-speech/parkinson1_train.txt", sep=",", header=None
    )
    ncols = df.shape[1] - 1
    X = np.asarray(df.iloc[:, 1 : ncols - 1], dtype=float)
    y = np.asarray(df.iloc[:, ncols])

    return X, y


def load_shd():
    df = pd.read_csv("datasets/statlog-heart/shd.dat", sep=" ", header=None)
    ncols = df.shape[1] - 1
    X = np.asarray(df.iloc[:, :ncols], dtype=int)
    y = np.where(df.iloc[:, ncols] == 2, 1, 0)

    return X, y


def load_tlcs():
    df = pd.read_csv("datasets/thoracic-surgery/tlcs.csv", sep=",", header=None)
    ncols = df.shape[1] - 1
    X = np.asarray(df.iloc[:, :ncols], dtype=float)
    y = np.asarray(df.iloc[:, ncols])

    return X, y


def load_vc():
    df = pd.read_csv("datasets/vertebral-column/vc.dat", sep=" ", header=None)
    ncols = df.shape[1] - 1
    X = np.asarray(df.iloc[:, :ncols], dtype=float)
    y = np.where(df.iloc[:, ncols] == "NO", 0, 1)

    return X, y
