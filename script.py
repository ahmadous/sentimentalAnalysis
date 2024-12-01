import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from werkzeug.utils import secure_filename
import os
import whisper  # Assurez-vous d'installer whisper : pip install whisper

from deepface import DeepFace
from PIL import Image
import pytesseract

# Télécharger les ressources nécessaires pour NLTK
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Charger le modèle de transcription Whisper
whisper_model = whisper.load_model("base")

# Configuration pour enregistrer temporairement les fichiers
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le pipeline d'analyse des sentiments
sentiment_analyzer = pipeline("sentiment-analysis")

# Télécharger les ressources nécessaires pour NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialiser Flask
app = Flask(__name__)
CORS(app)

# Configuration pour enregistrer temporairement les fichiers
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Charger le modèle d'analyse de sentiment
sentiment_analyzer = pipeline("sentiment-analysis")

# Fonction : Nettoyer le texte
def clean_text(text):
    if pd.isnull(text):
        return ""
    tokens = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words("english") + stopwords.words("french"))
    cleaned_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(cleaned_tokens)
# Fonction : Renommer les colonnes anonymes
def rename_columns(df):
    df.columns = [f"colonne_{i}" if not col or col.startswith("Unnamed") else col for i, col in enumerate(df.columns)]
    return df

@app.route("/analyze_media", methods=["POST"])
def analyze_media():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier audio ou vidéo envoyé."}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Transcrire le fichier avec Whisper
        transcription = whisper_model.transcribe(file_path)["text"]

        # Analyser le sentiment de la transcription
        sentiment = sentiment_analyzer(transcription)[0]
        label = sentiment["label"]
        score = sentiment["score"]

        # Suggestions en cas de sentiment négatif
        suggestions = []
        if label == "NEGATIVE":
            suggestions = [
                "Merci pour votre retour. Nous améliorerons nos services.",
                "Nous sommes désolés pour votre expérience. Veuillez nous contacter pour une assistance supplémentaire.",
            ]

        # Retourner les résultats
        return jsonify({
            "transcription": transcription,
            "sentiment": label,
            "score": score,
            "suggestions": suggestions,
        })
    except Exception as e:
        return jsonify({"error": f"Une erreur est survenue : {str(e)}"}), 500
    finally:
        # Supprimer le fichier temporaire
        os.remove(file_path)
@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    if "file" not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # Dictionnaire de traduction des émotions en français
        emotion_translation = {
            "angry": "Colère",
            "disgust": "Dégoût",
            "fear": "Peur",
            "happy": "Joie",
            "neutral": "Neutre",
            "sad": "Tristesse",
            "surprise": "Surprise",
        }

        # Tenter d'extraire du texte
        img = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(img).strip()

        if extracted_text:
            # Si du texte est trouvé, effectuer une analyse de sentiment
            sentiment = sentiment_analyzer(extracted_text)[0]
            label = sentiment["label"]
            score = sentiment["score"]
            return jsonify({
                "type": "text",
                "extracted_text": extracted_text,
                "sentiment": label,
                "score": score,
            })
        else:
            # Si aucun texte, effectuer une analyse faciale
            try:
                analysis = DeepFace.analyze(img_path=file_path, actions=["emotion"])
                dominant_emotion = emotion_translation[analysis[0]["dominant_emotion"]]
                emotion_scores = {
                    emotion_translation[key]: value
                    for key, value in analysis[0]["emotion"].items()
                }

                return jsonify({
                    "type": "face",
                    "dominant_emotion": dominant_emotion,
                    "emotion_scores": emotion_scores,
                })
            except Exception as e:
                return jsonify({"error": f"Erreur lors de l'analyse faciale : {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement de l'image : {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



# Fonction pour générer des suggestions
def generate_suggestions(sentiment_label):
    suggestions = {
        "POSITIVE": [
            "Merci pour votre retour positif ! 😊",
            "Nous sommes ravis que vous soyez satisfait. N'hésitez pas à partager votre expérience avec vos proches.",
        ],
        "NEGATIVE": [
            "Nous sommes désolés d'apprendre cela. Nous travaillons constamment à améliorer notre service.",
            "Pouvez-vous nous donner plus de détails sur votre expérience pour que nous puissions nous améliorer ?",
        ],
        "NEUTRAL": [
            "Merci pour votre retour !",
            "Si vous avez d'autres commentaires, n'hésitez pas à les partager avec nous.",
        ],
    }
    return suggestions.get(sentiment_label.upper(), ["Merci pour votre retour."])

@app.route("/predict", methods=["POST"])
def predict():
    if "text" not in request.json:
        return jsonify({"error": "Aucun texte fourni"}), 400

    text = request.json["text"]
    try:
        # Analyse des sentiments
        sentiment = sentiment_analyzer(text)[0]
        label = sentiment["label"]
        score = sentiment["score"]

        # Générer des suggestions
        suggestions = generate_suggestions(label)

        return jsonify({
            "text": text,
            "sentiment": label,
            "score": score,
            "suggestions": suggestions,
        })
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'analyse : {str(e)}"}), 500

# Fonction : Nettoyer les fichiers en renommant les colonnes anonymes
def clean_columns(dataframe):
    if not dataframe.columns[0].isidentifier():
        dataframe.columns = [f"colonne_{i}" for i in range(len(dataframe.columns))]
    return dataframe

# Fonction pour traiter les fichiers
def process_file(file_path):
    try:
        # Charger le fichier en fonction de son extension
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Format de fichier non pris en charge.")

        # Renommer les colonnes anonymes si nécessaire
        df = rename_columns(df)

        # Retourner les colonnes et un aperçu des 5 premières lignes
        columns = df.columns.tolist()
        preview = df.head(5).to_dict(orient="records")
        return df, columns, preview
    except Exception as e:
        return None, None, str(e)

# Route : Retourner les colonnes et l'aperçu d'un fichier

# Route : Extraire les colonnes et l'aperçu d'un fichier
@app.route("/columns", methods=["POST"])
def get_columns():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé."}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        df, columns, preview = process_file(file_path)
        if df is None:
            raise ValueError("Erreur lors du chargement du fichier.")

        return jsonify({"columns": columns, "preview": preview})
    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement du fichier : {str(e)}"}), 400
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



# Route : Analyser les sentiments
@app.route("/analyze", methods=["POST"])
def analyze_file():
    if "file" not in request.files or "text_column" not in request.form:
        return jsonify({"error": "Fichier ou colonne manquant."}), 400

    file = request.files["file"]
    text_column = request.form["text_column"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Charger le fichier et renommer les colonnes si nécessaire
        df, columns, preview = process_file(file_path)
        if df is None or text_column not in df.columns:
            raise ValueError("Colonne invalide ou fichier mal formaté.")

        # Prétraitement des données
        df[text_column] = df[text_column].fillna("").apply(clean_text)

        # Analyse des sentiments
        results = []
        sentiment_counts = {"Satisfait": 0, "Mécontent": 0, "Neutre": 0}
        for text in df[text_column]:
            sentiment = sentiment_analyzer(text)[0]
            sentiment_label = {
                "POSITIVE": "Satisfait",
                "NEGATIVE": "Mécontent",
                "NEUTRAL": "Neutre",
            }.get(sentiment["label"], "Inconnu")
            score = sentiment["score"]
            results.append({"Text": text, "Sentiment": sentiment_label, "Score": score})
            sentiment_counts[sentiment_label] += 1

        return jsonify({"results": results, "sentiment_counts": sentiment_counts})
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'analyse : {str(e)}"}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route("/analyze_folder", methods=["POST"])
def analyze_folder():
    folder_path = request.form.get("folder_path")
    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({"error": "Chemin de dossier invalide."}), 400

    results = []
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_path.endswith((".csv", ".xlsx", ".xls")):
                columns, preview = process_file(file_path)
                if columns:
                    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
                    for text in df[columns[0]].dropna():
                        sentiment = "Positive" if "good" in text.lower() else "Negative" if "bad" in text.lower() else "Neutral"
                        score = 0.9 if sentiment == "Positive" else 0.1 if sentiment == "Negative" else 0.5
                        results.append({"Text": text, "Sentiment": sentiment, "Score": score})
                        sentiment_counts[sentiment] += 1

        return jsonify({"results": results, "sentiment_counts": sentiment_counts})
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'analyse des fichiers : {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
