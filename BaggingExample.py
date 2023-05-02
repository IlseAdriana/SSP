from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def main():
    X, y = load_iris(return_X_y=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, stratify=y)

    knn_class = KNeighborsClassifier().fit(X_tr, y_tr)
    knn_pred = knn_class.predict(X_te)
    print(f'Accuracy KNN: {accuracy_score(knn_pred, y_te)}')

    bag_class = BaggingClassifier(
        base_estimator=KNeighborsClassifier(), n_estimators=3).fit(X_tr, y_tr)
    bag_pred = bag_class.predict(X_te)
    print(f'Accuracy Bagging: {accuracy_score(bag_pred, y_te)}')

if __name__ == '__main__':
    main()
