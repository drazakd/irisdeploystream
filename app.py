import streamlit as st
import pandas as pd
from model import charger_modele, faire_prediction

# Chargez le modèle entraîné (remplacez par votre chemin réel)
modele = charger_modele("classifier_model_iris.pkl")

# Structure de l'application avec les composants Streamlit
st.title("Application de classification d'iris")
st.header("Saisissez les caractéristiques de la fleur :")

# Récupération des données saisies par l'utilisateur pour les caractéristiques
# (adaptez aux caractéristiques de votre modèle)
longueur_sepale = st.number_input("Sepal.Length", min_value=0.0)
largeur_sepale = st.number_input("Sepal.Width", min_value=0.0)
longueur_petale = st.number_input("Petal.Length", min_value=0.0)
largeur_petale = st.number_input("Petal.Width", min_value=0.0)

# Création d'un DataFrame à partir des saisies de l'utilisateur (facultatif pour certains modèles)
data = pd.DataFrame({
    "Sepal.Length": [longueur_sepale],
    "Sepal.Width": [largeur_sepale],
    "Petal.Length": [longueur_petale],
    "Petal.Width": [largeur_petale]
})

# Prédiction au clic sur un bouton
if st.button("Prédire"):
    prediction = faire_prediction(modele, data)
    st.write(f"Espèce d'iris prédite : {prediction}")

st.stop()
