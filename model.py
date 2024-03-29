import joblib
import numpy as np  # Importer NumPy pour l'argument `argmax`
def charger_modele(chemin_modele):
    """Charge le modèle entraîné depuis le chemin spécifié."""
    try:
        return joblib.load(chemin_modele)
    except FileNotFoundError:
        raise Exception(f"Fichier modèle introuvable : {chemin_modele}")

def faire_prediction(modele, data):
    """Effectue une prédiction en utilisant le modèle de régression logistique chargé et retourne une valeur écrite."""

    # Prédiction des probabilités
    prediction = modele.predict_proba(data)

    # Conversion en valeur écrite basée sur la probabilité la plus élevée
    classe_predite = np.argmax(prediction, axis=1)[0]  # Index de la classe avec la plus haute probabilité

    if classe_predite == 0:
        return "Iris setosa"
    elif classe_predite == 1:
        return "Iris versicolor"
    else:
        return "Iris virginica"  # En supposant que la classe 2 correspond à Iris virginica

