import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # pour sauvegarder le modèle

# Charger les données prétraitées (tu peux aussi utiliser tes propres fonctions de prétraitement ici)
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Prétraiter les données (assume que tu as déjà une fonction de prétraitement)
def preprocess_data(df):
    # Ajoute ici tes étapes de prétraitement si nécessaires
    return df

# Charger et préparer les données
df = load_data("C:/Users/ekavu/Angelikia Kavuansiko/projet-titanic/projet_ingenieurie/data/preprocessed_train.csv")  # Remplace ce chemin par celui où tes données prétraitées sont stockées
df = preprocess_data(df)

# Séparer les caractéristiques et les labels
X = df.drop("Survived", axis=1)  # "Survived" est l'étiquette
y = df["Survived"]

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer la précision du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Sauvegarder le modèle dans un fichier
joblib.dump(model, "trained_model.pkl")
print("Model saved as trained_model.pkl")
