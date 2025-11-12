# ce code sert à titre d'exemple pour quand plusieurs colonnes sont importantes pour le résultat final:
#############################################
######## CLASSIFICATION MULTI OUTPUT ########
#############################################

# input	x	y
# Image1	45	80
# Image2	60	100

# ➡️ Ici :

# X = image (ou ses caractéristiques)

# y = 2 colonnes (x, y)
# X = df.drop(columns=["x", "y"])
# y = df[["x", "y"]]  # Plusieurs colonnes de sortie

# from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor()
# model.fit(X, y)
# pred = model.predict(X_test)

############################################
######## CLASSIFICATION MULTI LABEL ########
############################################

# Un film peut avoir plusieurs genres à la fois :

# Titre	Action	Comédie	Drame
# Avengers	1	1	0
# Titanic	0	0	1
# Deadpool	1	1	0

# ➡️ Tu veux donc prédire plusieurs 0/1 en même temps.
# C’est une classification multi-label.

# En code :

# X = df[["duration", "budget", "rating"]]  # caractéristiques
# y = df[["Action", "Comedy", "Drama"]]      # plusieurs colonnes à prédire

# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import RandomForestClassifier

# model = MultiOutputClassifier(RandomForestClassifier())
# model.fit(X, y)


# Résultat : le modèle prédit un vecteur [1, 0, 1] pour chaque film
# → (Action = oui, Comédie = non, Drame = oui).

##############################################
######## CLASSIFICATION MULTI Valeurs ########
##############################################

# âge	revenu	satisfaction	probabilité_d’achat
# 25	1800	4.3	0.75

# Tu veux prédire :

# revenu → numérique (régression)

# satisfaction → numérique (régression)

# probabilité_d’achat → entre 0 et 1 (classification)

# ➡️ Il faut alors plusieurs modèles distincts, un pour chaque type de variable :

# model_revenu = RandomForestRegressor()
# model_satisfaction = RandomForestRegressor()
# model_achat = RandomForestClassifier()

# model_revenu.fit(X, y["revenu"])
# model_satisfaction.fit(X, y["satisfaction"])
# model_achat.fit(X, y["probabilité_d’achat"])