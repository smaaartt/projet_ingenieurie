#Importation des bibliothéques 
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.model_training import load_data, split_data, train_model, save_model

# Étape de tests
class TestModelTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Création d'un DataFrame de test"""
        cls.test_data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [6, 7, 8, 9, 10],
            'Survived': [0, 1, 1, 0, 1]
        })
        cls.test_file = "test_data.csv"
        cls.model_file = "test_model.pkl"

        cls.test_data.to_csv(cls.test_file, index=False)  # Créer un fichier CSV pour les tests
    """
    cls.test_data : Création d'un DataFrame avec des données fictives
    cls.test_file : Chemin du fichier CSV où les données seront sauvegardée.
    cls.model_file : Chemin du fichier où le modèle sera sauvegardé
    to_csv(cls.test_file, index=False) : Sauvegarde des données dans un fichier CSV (avec Index)
    """
    
    
    def test_load_data(self):
        """Test du chargement des données"""
        df = load_data(self.test_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, self.test_data.shape)
    """
    load_data() : Charge les données depuis le fichier CSV
    assertIsInstance() : Vérifie que df est bien un DataFrame
    assertEqual() : Vérifie que la forme des données chargées est identique à celle du DataFrame de test (self.test_data)
    """


    def test_split_data(self):
        """Test de la séparation des données"""
        df = load_data(self.test_file)
        X_train, X_test, y_train, y_test = split_data(df)

        self.assertEqual(len(X_train) + len(X_test), len(df))
        self.assertEqual(len(y_train) + len(y_test), len(df))
        self.assertGreater(len(X_train), len(X_test))
    """
    split_data(df) : Sépare les données en variables d'entrée (X_train, X_test) et en variable cible (y_train, y_test).
assertEqual() : Vérifie que la somme des tailles de X_train et X_test est égale au nombre total d'exemples dans df.
assertEqual() : Vérifie que la somme des tailles de y_train et y_test est égale au nombre total d'exemples dans df.
assertGreater() : Vérifie que la taille de l'ensemble d'entraînement est plus grande que celle de l'ensemble de test (80/20 ou similaire)
    """


    def test_train_model(self):
        """Test de l'entraînement du modèle"""
        df = load_data(self.test_file)
        X_train, X_test, y_train, y_test = split_data(df)

        model = train_model(X_train, y_train)

        self.assertIsInstance(model, RandomForestClassifier)
        self.assertTrue(hasattr(model, "predict"))
    """
    train_model( : Entraîne un modèle RandomForestClassifier sur les données d'entraînement.
    assertIsInstance() : Vérifie que le modèle est bien une instance de RandomForestClassifier.
    assertTrue() : Vérifie que le modèle possède la méthode predict (prédictions)
    """


    def test_save_model(self):
        """Test de la sauvegarde du modèle"""
        df = load_data(self.test_file)
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_model(X_train, y_train)

        save_model(model, self.model_file)
        self.assertTrue(os.path.exists(self.model_file))
    """
    save_model() : Sauvegarde le modèle dans un fichier.
    assertTrue() : Vérifie que le fichier contenant le modèle existe après l'exécution de save_model
    """


    @classmethod
    def tearDownClass(cls):
        """Suppression des fichiers de test"""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)
        if os.path.exists(cls.model_file):
            os.remove(cls.model_file)
    """
    os.remove() et os.remove() : Supprime les fichiers temporaires utilisés pendant les tests (test_data.csv et test_model.pkl)
    """        
            
    
if __name__ == "__main__":
    unittest.main()
"""
Exécute  les tests si le fichier est exécuté directement, plutôt que d'être importé comme module
unittest.main() découvre et exécute toutes les méthodes de test définies dans la classe TestModelTraining
"""    

