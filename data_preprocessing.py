import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Charge les fichiers train et test."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test




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

def split_data(df):
    """Sépare les données en ensembles d'entraînement et de validation."""
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    train, test = load_data("data/train.csv", "data/test.csv")
    print("🔹 Données chargées avec succès !")
    
    train = preprocess_data(train)
    print("🔹 Prétraitement terminé !")

    output_path = "data/processed_train.csv"
    train.to_csv(output_path, index=False)

    print(f"✅ Fichier sauvegardé : {output_path}")
