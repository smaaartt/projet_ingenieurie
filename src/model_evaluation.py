import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Charger les données de test
def load_test_data(test_path):
    """Charge les données de test prétraitées"""
    df = pd.read_csv(test_path)
    return df

# Charger le modèle entraîné
def load_model(model_path):
    """Charge le modèle sauvegardé"""
    model = joblib.load(model_path)
    return model

# Évaluer le modèle
def evaluate_model(model, X_test, y_test):
    """Fait des prédictions et calcule la précision"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"🎯 Précision du modèle : {accuracy:.2f}")
    print("\n📊 Rapport de classification :\n", report)

if __name__ == "__main__":
    print("🔹 Chargement des données de test...")
    df_test = load_test_data("data/processed_train.csv")  # ⚠️ À adapter si test.csv

    print("🔹 Séparation des features et de la cible...")
    X_test = df_test.drop(columns=['Survived'])
    y_test = df_test['Survived']

    print("🔹 Chargement du modèle...")
    model = load_model("models/random_forest.pkl")

    print("🔹 Évaluation du modèle...")
    evaluate_model(model, X_test, y_test)
