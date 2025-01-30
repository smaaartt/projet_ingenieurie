import pandas as pd
import joblib  # Pour sauvegarder le modèle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données prétraitées
def load_data(file_path):
    """Charge les données nettoyées"""
    df = pd.read_csv(file_path)
    return df

# Séparer les features et la cible
def split_data(df):
    """Sépare les features (X) et la variable cible (y)"""
    X = df.drop(columns=['Survived'])  # Supprime la colonne cible
    y = df['Survived']  # Cible (0 = Non, 1 = Oui)
    return train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

# Entraîner le modèle
def train_model(X_train, y_train):
    """Entraîne un RandomForestClassifier"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Sauvegarder le modèle
def save_model(model, model_path):
    """Sauvegarde le modèle entraîné"""
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé : {model_path}")

if __name__ == "__main__":
    print("🔹 Chargement des données...")
    df = load_data("data/processed_train.csv")

    print("🔹 Séparation des données...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("🔹 Entraînement du modèle...")
    model = train_model(X_train, y_train)

    print("🔹 Évaluation du modèle...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Précision du modèle : {accuracy:.2f}")

    print("🔹 Sauvegarde du modèle...")
    save_model(model, "models/random_forest.pkl")
