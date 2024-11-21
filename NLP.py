import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Charger le modèle et le vectoriseur TF-IDF
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialisation des objets nécessaires pour le prétraitement
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fonction de prétraitement des tweets
def preprocess_text(text):
    text = text.lower()  # Convertir en minuscules
    text = text.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
    tokens = [word for word in text.split() if word not in stop_words]  # Retirer les stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatisation
    return ' '.join(tokens)

# Interface Streamlit
st.title("Détection de Tweets Suspects")
st.write("Ce modèle permet de classer un tweet comme **suspect** ou **non suspect**.")

# Saisie du tweet par l'utilisateur
new_tweet = st.text_area("Saisissez un tweet à analyser :")

# Bouton pour lancer la prédiction
if st.button("Analyser"):
    if new_tweet:
        # Prétraiter le tweet
        cleaned_tweet = preprocess_text(new_tweet)
        
        # Transformer le tweet avec le vectoriseur TF-IDF
        tweet_vector = vectorizer.transform([cleaned_tweet])
        
        # Faire la prédiction
        prediction = model.predict(tweet_vector)
        
        # Afficher le résultat
        if prediction == 1:
            st.error("Le tweet est suspect (menaces, terrorisme, intimidation...).")
        else:
            st.success("Le tweet n'est pas suspect.")
    else:
        st.warning("Veuillez entrer un tweet à analyser.")
