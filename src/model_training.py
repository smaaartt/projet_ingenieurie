"""
Module d'entraînement du modèle pour le projet Titanic.

Ce script charge les données prétraitées, entraîne un modèle de classification 
(RandomForestClassifier), évalue sa précision et sauvegarde le modèle entraîné.
"""

import pandas as pd
import joblib  # Pour sauvegarder le modèle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Charge les données nettoyées à partir d'un fichier CSV.

    Args:
        file_path (str): Chemin vers le fichier contenant les données prétraitées.

    Returns:
        pd.DataFrame: DataFrame contenant les données nettoyées.
    """
    return pd.read_csv(file_path)

def split_data(df):
    """
    Sépare les données en features (X) et variable cible (y), puis divise en train/test.

    Args:
        df (pd.DataFrame): Données prétraitées contenant la colonne 'Survived'.

    Returns:
        tuple: X_train, X_test, y_train, y_test (ensembles d'entraînement et de test).
    """
    X = df.drop(columns=['Survived'])  # Supprime la colonne cible
    y = df['Survived']  # Cible (0 = Non, 1 = Oui)
    return train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

def train_model(X_train, y_train):
    """
    Entraîne un modèle RandomForestClassifier sur les données d'entraînement.

    Args:
        X_train (pd.DataFrame): Features d'entraînement.
        y_train (pd.Series): Labels d'entraînement.

    Returns:
        RandomForestClassifier: Modèle entraîné.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """
    Sauvegarde le modèle entraîné dans un fichier .pkl.

    Args:
        model (RandomForestClassifier): Modèle entraîné.
        model_path (str): Chemin du fichier où sauvegarder le modèle.
    """
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé : {model_path}")

if __name__ == "__main__":
    print("🔹 Chargement des données...")
    df = load_data("data/processed_train.csv")

    print("🔹 Séparation des données en features et labels...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("🔹 Entraînement du modèle...")
    model = train_model(X_train, y_train)

    print("🔹 Évaluation du modèle...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Précision du modèle : {accuracy:.2f}")

    print("🔹 Sauvegarde du modèle entraîné...")
    save_model(model, "models/random_forest.pkl")
