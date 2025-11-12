from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger les données
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un pipeline : normalisation + modèle
pipeline = make_pipeline(
    StandardScaler(),          # Étape 1 : mettre les données à la même échelle
    RandomForestClassifier()   # Étape 2 : modèle
)

# Entraîner le pipeline
pipeline.fit(X_train, y_train)

# Prédire
y_pred = pipeline.predict(X_test)

# Évaluer
print("Accuracy avec pipeline :", accuracy_score(y_test, y_pred))
