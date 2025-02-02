# main.py
import subprocess

def run_script(script_name):
    """Exécute un script Python en utilisant subprocess."""
    try:
        print(f"Exécution de {script_name}...")
        subprocess.run(["python", script_name], check=True)
        print(f"{script_name} terminé avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {script_name}: {e}")
        raise

if __name__ == "__main__":
    # Liste des scripts à exécuter dans l'ordre
    scripts = [
        "src/data_preprocessing.py",
        "src/model_training.py",
        "src/model_evaluation.py"
    ]

    # Exécution des scripts
    for script in scripts:
        run_script(script)

    print("Tous les scripts ont été exécutés avec succès.")