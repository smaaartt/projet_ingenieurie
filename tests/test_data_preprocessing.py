# Importation des libraries
import pytest
import pandas as pd
from io import StringIO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.data_preprocessing import load_data, preprocess_data, split_data

pytest.main(["C:/projet-titanic/tests/TestUnit.py"])


# Données simulées en format CSV
MOCK_CSV = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,Name1,male,22,1,0,A/5 21171,7.25,,S
2,1,1,Name2,female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,Name3,female,,0,0,STON/O2. 3101282,7.925,,S
"""

@pytest.fixture
def mock_dataframe():
    """Créer un DataFrame simulé à partir de données CSV."""
    return pd.read_csv(StringIO(MOCK_CSV))
"""
@pytest.fixture : Permet de réutiliser un objet dans plusieurs tests.
mock_dataframe() retourne un DataFrame pandas à partir du CSV simulé.
StringIO(MOCK_CSV) transforme le texte en fichier virtuel pour pd.read_csv().
"""


# Test de load_data (simulation en mémoire)
def test_load_data(mocker):
    """Teste si load_data charge correctement le fichier CSV"""
    mocker.patch("pandas.read_csv", return_value=pd.read_csv(StringIO(MOCK_CSV)))
    train, test = load_data("fake_path_train.csv", "fake_path_test.csv")
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert not train.empty
    assert not test.empty
"""
mocker.patch(...)
→ Simule pd.read_csv() pour qu'il retourne notre DataFrame sans lecture de vrai fichier.
Appel de load_data("fake_path_train.csv", "fake_path_test.csv")
→ Même si les fichiers "fake_path_train.csv" et "fake_path_test.csv" n'existent pas, la fonction retournera notre DataFrame simulé.
Assertions (assert) :
Vérifie que train et test sont des DataFrame.
Vérifie qu'ils ne sont pas vides (not train.empty).
"""


# Test de preprocess_data
def test_preprocess_data(mock_dataframe):
    """Teste la fonction preprocess_data."""
    df = preprocess_data(mock_dataframe)
    assert 'Age' in df.columns and df['Age'].isnull().sum() == 0  # Vérifie que les NaN sont remplis
    assert 'Embarked_S' in df.columns  # Vérifie que l'encodage est bien fait
    assert 'Cabin' not in df.columns  # Vérifie la suppression des colonnes
    assert 'Name' not in df.columns  # Vérifie la suppression de 'Name'
    assert 'PassengerId' not in df.columns  # Vérifie la suppression de 'PassengerId'
"""
df = preprocess_data()
→ Transformation du DataFrame en appliquant la fonction preprocess_data().
Assert:
Vérifie que la colonne "Age" est toujours présente et qu'il n'y a plus de valeurs manquantes.
Vérifie que "Embarked_S" a été ajouté.
Vérifie que "Cabin", "Name" et "PassengerId" ont été supprimés.
"""


# Test de split_data
def test_split_data(mock_dataframe):
    """Teste la fonction split_data."""
    df = preprocess_data(mock_dataframe)
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) > 0 and len(X_test) > 0  # Vérifie que le split fonctionne bien
    assert len(y_train) > 0 and len(y_test)
"""
Transformation des données : df = preprocess_data(mock_dataframe)
Séparation des données : X_train, X_test, y_train, y_test = split_data(df)
Assertions (assert) :
Vérifie que X_train et X_test contiennent des données.
Vérifie que y_train et y_test contiennent des labels (Survived).
"""
