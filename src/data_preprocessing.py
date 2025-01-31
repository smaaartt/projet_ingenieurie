# Importation des bibliothéques
import pandas as pd
from sklearn.model_selection import train_test_split


# Importation des données
def load_data(train_path, test_path):
    """Charge les fichiers train et test."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
"""
Fonction prennant en entrée deux chemins de fichiers (csv).
Utilisation pd.read_csv() pour lire les fichiers CSV et les stocker sous forme de DataFrame.
Retourne deux DataFrames : train et test.
"""


def preprocess_data(df):
    """Nettoie et prépare les données pour l'entraînement du modèle."""
    df = df.copy()
    
    # Remplissage des valeurs manquantes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df.loc[:, 'Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Suppression des colonnes inutiles
    df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
    
    # Encodage des variables catégoriques
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    return df
"""
Copie du dataframe pour éviter de modifier les données originales directement.

variable Age : Remplace les valeurs manquantes par la médiane.
variable Embarked : Remplace les valeurs manquantes par la valeur la plus fréquente.
variable Fare → Remplace les valeurs manquantes par la médiane.

Suppression des variables Cabin, Ticket, Name, PassengerId.

Transformation des variables catégoriques (Sex, Embarked) en variables numériques (0 ou 1).
Permet d'éviter la multicolinéarité (éviter d’avoir des informations redondantes).
"""


def split_data(df):
    """Sépare les données en ensembles d'entraînement et de validation."""
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)
"""
Variable x contient toutes les colonnes sauf Survived.
Variable y contient uniquement la colonne Survived (prédiction).
Division en train/test
	0.2 → 20% des données seront mises de côté pour les tests.
	42 → Assure que la répartition est toujours la même pour avoir des résultats reproductibles.
"""


if __name__ == "__main__":
    train, test = load_data("C:/projet-titanic/data/train.csv", "C:/projet-titanic/data/test.csv")
    print("🔹 Données chargées avec succès !")
    
    train = preprocess_data(train)
    print("🔹 Prétraitement terminé !")

    output_path = "C:/projet-titanic/data/processed_train.csv"
    train.to_csv(output_path, index=False)

    print(f"✅ Fichier sauvegardé : {output_path}")
"""
Si les données train et test sont chargées → retourne un message indiquant que le chargement est effectué.
Application de la fonction 'preprocess_data' sur les données train → retourne un message indiquant que le traitement est effectué.
Sauvegarde des données issues du prétraitement dans un fichier csv → retourne un message indiquant que l'enregistrement est effectué.
"""