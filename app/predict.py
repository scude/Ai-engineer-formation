# Fonctions de prédiction utilisées par l’API
def predict_sentiment(text: str) -> str:
    # Dummy implémentation pour test
    return "positive" if "good" in text.lower() else "negative"
