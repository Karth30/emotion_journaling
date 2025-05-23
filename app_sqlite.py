import joblib
import numpy as np
import sqlite3
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import streamlit as st
import matplotlib.pyplot as plt

# Load model and utilities
model = tf.keras.models.load_model("model/biLSTM+CNN.keras")
tokenizer = joblib.load("model/tokenizer.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

# Parameters
max_len = 100
confidence_threshold = 0.4

# SQLite Database Setup
def init_db():
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL)''')

    c.execute('''CREATE TABLE IF NOT EXISTS journals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    journal TEXT,
                    hard_emotion TEXT,
                    soft_emotion TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# Text preprocessing function
def preprocess_text(text):
    words = re.findall(r"[\w']+|[.,!?;]", text)
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return " ".join(words)

# Sentence splitter with improved regex
def split_sentences(text):
    return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|\n)\s+', text)

# Emotion prediction function
def predict_emotion(sentence):
    processed_sentence = preprocess_text(sentence)
    sequence = tokenizer.texts_to_sequences([processed_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")
    prediction = model.predict(padded_sequence)[0]

    max_prob = np.max(prediction)
    predicted_label = np.argmax(prediction)
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_emotion if max_prob >= confidence_threshold else "Neutral", prediction

# Journal analysis function
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

    # Hard Prediction (Most Frequent Emotion)
    emotion_counts = Counter(emotion for _, emotion in sentence_emotions)
    dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "Neutral"

    # Soft Prediction (Mean Probabilities)
    if np.sum(emotion_scores) == 0:
        soft_emotion = "Neutral"
    else:
        soft_emotion = label_encoder.inverse_transform([np.argmax(emotion_scores)])[0]

    return sentence_emotions, dominant_emotion, soft_emotion, emotion_scores

# User Authentication
def register_user(username, password):
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

# Store Journal Entry
def store_journal(user_id, journal, hard_emotion, soft_emotion):
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("INSERT INTO journals (user_id, journal, hard_emotion, soft_emotion) VALUES (?, ?, ?, ?)",
              (user_id, journal, hard_emotion, soft_emotion))
    conn.commit()
    conn.close()

# Retrieve Previous Journals
def get_previous_journals(user_id):
    conn = sqlite3.connect("user_data.db")
    c = conn.cursor()
    c.execute("SELECT journal, hard_emotion, soft_emotion, timestamp FROM journals WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    journals = c.fetchall()
    conn.close()
    return journals

# Main Streamlit App
def main():
    st.title("📝 Emotion Detection in Journals")

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    # Authentication
    if st.session_state.user_id is None:
        option = st.radio("Select an option:", ("Login", "Register"))

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Submit"):
            if option == "Register":
                if register_user(username, password):
                    st.success("Registration successful! Please log in.")
                else:
                    st.error("Username already exists.")
            else:
                user_id = authenticate_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    # Journal Entry
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
            st.write(f"**Hard Prediction (Most Frequent Emotion):** {dominant_emotion}")
            st.write(f"**Soft Prediction (Most Probable Emotion):** {soft_emotion}")

            st.subheader("📊 Emotion Distribution")

            # Create figure and axis with better size and clarity
            fig, ax = plt.subplots(figsize=(10, 6))

            # Use a vibrant color palette for better visual impact
            colors = plt.cm.Paired(np.linspace(0, 1, len(label_encoder.classes_)))

            # Enhanced bar chart with styling
            ax.bar(label_encoder.classes_, emotion_scores, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

            # Add labels and title with improved styling
            ax.set_ylabel("Probability", fontsize=12, fontweight='bold')
            ax.set_xlabel("Emotions", fontsize=12, fontweight='bold')
            ax.set_title("🌟 Emotion Distribution Across Journal", fontsize=16, fontweight='bold')

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, fontsize=10)

            # Add a grid for clarity
            ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

            # Add data labels on top of the bars for better insight
            for i, v in enumerate(emotion_scores):
                ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10, fontweight='bold')

            # Set background for better contrast
            fig.patch.set_facecolor('#F5F5F5')
            ax.set_facecolor('#FAFAFA')

            # Display the improved plot
            st.pyplot(fig)


        if st.button("View Previous Journals"):
            journals = get_previous_journals(st.session_state.user_id)
            dates = []
            emotions = []
            for journal, hard_emotion, soft_emotion, timestamp in journals:
                st.write(f"📅 {timestamp}: **{hard_emotion}** | {soft_emotion}")
                st.write(f"📝 {journal}")
                dates.append(timestamp)
                emotions.append(soft_emotion)

            if dates and emotions:
                st.subheader("📈 Emotion Trend Over Time")
                fig, ax = plt.subplots(figsize=(10, 6))

                # Gradient-like color for better visualization
                colors = plt.cm.viridis(np.linspace(0, 1, len(dates)))

                # Plot with improved styling
                ax.scatter(dates[::-1], emotions[::-1], c=colors, edgecolor='black', linewidth=1.2, s=100, label="Emotion Points")
                ax.plot(dates[::-1], emotions[::-1], linestyle='--', color='green', alpha=0.7, linewidth=2, label="Emotion Trend")

                # Customize labels and title
                ax.set_xlabel("Date", fontsize=12, fontweight='bold')
                ax.set_ylabel("Soft Emotion", fontsize=12, fontweight='bold')
                ax.set_title("💚 Your Emotion Over Time", fontsize=16, fontweight='bold')

                # Improve x-axis readability
                plt.xticks(rotation=45, fontsize=10)
                ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

                # Add a legend for clarity
                ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor="black")

                # Add a background color
                fig.patch.set_facecolor('#F5F5F5')
                ax.set_facecolor('#FAFAFA')                                     

                # Display the plot
                st.pyplot(fig)


if __name__ == '__main__':
    main()
