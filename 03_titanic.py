import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = sns.load_dataset("titanic")
print(df.shape)

# 01 Nettoyage des données
df = df.drop(columns=["deck", "embark_town"])  # Supprime les colonnes peu utiles
df["age"] = df["age"].fillna(df["age"].median())  # Remplace les NaN par la médiane
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])  # Remplace NaN par la valeur la plus fréquente

# 02 Séparer Feature et target
X = df.drop(columns=["survived"])  # Doit deviner survived à partir des autres colonnes
y = df["survived"]                 # Cible (0 = mort, 1 = survécu)

# 03 Identifier les types de variables
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns
#les données triées
print("Numériques :", list(num_features))
print("Catégorielles :", list(cat_features))

# 04 Création d'un pipeline mixte pour combiner num et catégorielles
num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

# Pipeline complet : prétraitement + modèle
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy :", model.score(X_test, y_test))

