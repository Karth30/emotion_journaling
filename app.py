import joblib
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from supabase import create_client, Client
import bcrypt
import nltk

# Load model and utilities
model = tf.keras.models.load_model("model/biLSTM.keras")
tokenizer = joblib.load("model/tokenizer.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

# Parameters
max_len = 100
confidence_threshold = 0.3

# Connect to Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Text preprocessing
def preprocess_text(text):
    words = re.findall(r"[\w']+|[.,!?;]", text)
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return " ".join(words)

# Split sentences
def split_sentences(text):
    # Handle various sentence-ending punctuation and line breaks
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*', text.strip())

# Predict emotion
def predict_emotion(sentence):
    processed_sentence = preprocess_text(sentence)
    sequence = tokenizer.texts_to_sequences([processed_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")
    prediction = model.predict(padded_sequence)[0]

    max_prob = np.max(prediction)
    predicted_label = np.argmax(prediction)
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_emotion if max_prob >= confidence_threshold else "Neutral", prediction

# Analyze journal
def analyze_journal(journal):
    sentences = split_sentences(journal)
    sentence_emotions = []
    emotion_scores = np.zeros(len(label_encoder.classes_))

    for sentence in sentences:
        if not sentence.strip():
            continue
        emotion, probabilities = predict_emotion(sentence)
        if emotion != "Neutral":
            sentence_emotions.append((sentence, emotion))
            emotion_scores += probabilities

    emotion_counts = Counter(emotion for _, emotion in sentence_emotions)
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Neutral"

    soft_emotion = "Neutral" if np.sum(emotion_scores) == 0 else label_encoder.inverse_transform([np.argmax(emotion_scores)])[0]
    return sentence_emotions, dominant_emotion, soft_emotion, emotion_scores

# Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Verify password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

# User Registration
def register_user(username, password):
    try:
        hashed_password = hash_password(password)

        response = supabase.table("users").insert({
            "username": username,
            "password": hashed_password
        }).execute()
        return response.data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# User Authentication
def authenticate_user(username, password):
    try:
        response = supabase.table("users").select("id, password").eq("username", username).execute()
        if response.data and verify_password(password, response.data[0]["password"]):
            return response.data[0]["id"]
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Store Journal
def store_journal(user_id, journal, hard_emotion, soft_emotion):
    try:
        supabase.table("journals").insert({
            "user_id": user_id,
            "journal": journal,
            "hard_emotion": hard_emotion,
            "soft_emotion": soft_emotion
        }).execute()
    except Exception as e:
        st.error(f"Error: {e}")

# Retrieve Previous Journals
def get_previous_journals(user_id):
    try:
        response = supabase.table("journals").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Error: {e}")
        return []

# Main App
def main():
    st.title("📝 Emotion Detection in Journals")

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if st.session_state.user_id is None:
        option = st.radio("Select an option:", ("Login", "Register"))
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Submit"):
            if option == "Register":
                if register_user(username, password):
                    st.success("Registration successful! Please log in.")
                else:
                    st.error("Username already exists or error occurred.")
            else:
                user_id = authenticate_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    else:
        st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())
        journal = st.text_area("Enter your journal entry below:", height=300)

        if st.button("Analyze Emotions"):
            if not journal.strip():
                st.warning("Please enter a journal entry to analyze.")
                return

            sentence_emotions, dominant_emotion, soft_emotion, emotion_scores = analyze_journal(journal)
            store_journal(st.session_state.user_id, journal, dominant_emotion, soft_emotion)

            st.subheader("📌 Sentence-wise Emotion Predictions")
            for sentence, emotion in sentence_emotions:
                st.write(f"➡ **{sentence}** → _{emotion}_")

            st.subheader("🧠 Overall Emotion Analysis")
            st.write(f"**Hard Emotion:** {dominant_emotion}")
            st.write(f"**Soft Emotion:** {soft_emotion}")

            st.subheader("📊 Emotion Distribution")
            fig, ax = plt.subplots()
            ax.bar(label_encoder.classes_, emotion_scores, color="skyblue")
            st.pyplot(fig)

        if st.button("View Previous Journals"):
            journals = get_previous_journals(st.session_state.user_id)
            dates = []
            emotions = []
            for entry in journals:
                st.write(f"📅 {entry['timestamp']}: **{entry['hard_emotion']}** | {entry['soft_emotion']}")
                st.write(f"📝 {entry['journal']}")
                dates.append(entry['timestamp'])
                emotions.append(entry['soft_emotion'])

            if dates and emotions:
                st.subheader("📈 Emotion Trend Over Time")
                fig, ax = plt.subplots()
                ax.plot(dates[::-1], emotions[::-1], marker='o', linestyle='-', color='green')
                ax.set_xlabel("Date")
                ax.set_ylabel("Soft Emotion")
                ax.set_title("Your Emotion Over Time")
                plt.xticks(rotation=45)
                st.pyplot(fig)

if __name__ == '__main__':
    main()