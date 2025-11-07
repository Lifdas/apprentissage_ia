from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame

print(df.head())        #affiche les 5 premières valeurs
print(df.describe())  # statistiques de base comme moyenne, min-max std(écart type)
print(df.info())      # types et valeurs manquantes (Nan)

# 1. Charger les données
X, y = load_iris(return_X_y=True)

# 2. Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Définir les modèles
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighbors": KNeighborsClassifier()
}

# 4. Boucle pour entraîner et évaluer
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")


