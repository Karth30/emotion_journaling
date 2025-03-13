import joblib
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re
import streamlit as st
import matplotlib.pyplot as plt

# Load model and utilities
model = tf.keras.models.load_model("model/biLSTM.keras")
tokenizer = joblib.load("model/tokenizer.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

# Parameters
max_len = 100
confidence_threshold = 0.3

# Text preprocessing function
def preprocess_text(text):
    words = re.findall(r"[\w']+|[.,!?;]", text)
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return " ".join(words)

# Sentence splitter with improved regex
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

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

# Main Streamlit app
def main():
    st.title("📝 Emotion Detection in Journals")

    journal = st.text_area("Enter your journal entry below:", height=300)

    if st.button("Analyze Emotions"):
        if not journal.strip():
            st.warning("Please enter a journal entry to analyze.")
            return

        sentence_emotions, dominant_emotion, soft_emotion, emotion_scores = analyze_journal(journal)

        # Display sentence-wise predictions
        st.subheader("📌 Sentence-wise Emotion Predictions")
        for sentence, emotion in sentence_emotions:
            st.write(f"➡ **{sentence}** → _{emotion}_")

        # Display dominant emotions
        st.subheader("🧠 Overall Emotion Analysis")
        st.write(f"**Hard Prediction (Most Frequent Emotion):** {dominant_emotion}")
        st.write(f"**Soft Prediction (Most Probable Emotion):** {soft_emotion}")

        # Plot emotion distribution
        st.subheader("📊 Emotion Distribution")
        fig, ax = plt.subplots()
        ax.bar(label_encoder.classes_, emotion_scores, color="skyblue")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotions")
        ax.set_title("Emotion Distribution Across Journal")
        st.pyplot(fig)

if __name__ == '__main__':
    main()
