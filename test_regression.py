import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 1. Créer un dataset simple (2 features, 2 classes)
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=42
)

# 2. Entraîner un modèle
model = LogisticRegression()
model.fit(X, y)

# 3. Créer une grille de points pour visualiser l’espace
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 4. Prédire pour chaque point de la grille
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Tracer la frontière de décision
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k', cmap='coolwarm')
plt.title("Frontière de décision – Régression Logistique")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
