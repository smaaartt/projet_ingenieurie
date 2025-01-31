"""
Module d'évaluation du modèle pour le projet Titanic.

Ce script charge le modèle entraîné et les données de test, effectue des prédictions 
et affiche des métriques d'évaluation (accuracy, classification report).
"""

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def load_test_data(test_path):
    """
    Charge les données de test prétraitées.

    Args:
        test_path (str): Chemin du fichier CSV des données test.

    Returns:
        pd.DataFrame: Données de test sous forme de DataFrame.
    """
    return pd.read_csv(test_path)

def load_model(model_path):
    """
    Charge le modèle entraîné depuis un fichier.

    Args:
        model_path (str): Chemin du fichier modèle (.pkl).

    Returns:
        RandomForestClassifier: Modèle chargé.
    """
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur des données de test en affichant :
    - La précision globale (accuracy)
    - Le rapport de classification détaillé (precision, recall, f1-score)

    Args:
        model (RandomForestClassifier): Modèle entraîné.
        X_test (pd.DataFrame): Features de test.
        y_test (pd.Series): Labels de test.

    Prints:
        - Accuracy du modèle.
        - Rapport de classification.
    """
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
