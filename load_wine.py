from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline avec scaling + logreg (plus d'itérations)
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver='lbfgs')
)

pipeline.fit(X_train, y_train)

print("Train accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy: ", pipeline.score(X_test, y_test))

# Cross-validation pour une évaluation plus robuste
#découpe en 5 blocs le dataset de train, et fait 1 bloc + un peu de test et compare si train = test sur les 5 blocs
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv)
print("CV mean accuracy:", np.mean(cv_scores), " std:", np.std(cv_scores))
