import joblib
import numpy as np
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.utils import simple_preprocess
import tensorflow as tf

# Load model, tokenizer, and label encoder
model = tf.keras.models.load_model("model/biLSTM.keras")
tokenizer = joblib.load("model/tokenizer.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Define max sequence length (same as training)
max_len = 100  
confidence_threshold = 0.6  # Only consider emotions above this probability


def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize input text."""
    words = simple_preprocess(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


def predict_emotion(sentence):
    """Predict emotion and return the emotion with confidence score."""
    processed_sentence = preprocess_text(sentence)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([processed_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")

    # Make prediction
    prediction = model.predict(padded_sequence)[0]  # Get probability distribution

    # Get the most probable emotion
    max_prob = np.max(prediction)
    predicted_label = np.argmax(prediction)
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

    # Return emotion only if confidence is above the threshold
    if max_prob >= confidence_threshold:
        return predicted_emotion
    else:
        return "Neutral"  # Ignore low-confidence predictions


# User Input
if __name__ == "__main__":
    emotions = defaultdict(int)
    journal = input("Enter your journal entry: ")

    # Tokenize into sentences
    sentences = sent_tokenize(journal)

    # Predict emotions for each sentence
    for sentence in sentences:
        emotion = predict_emotion(sentence)
        if emotion != "Neutral":  # Ignore sentences with no strong emotion
            emotions[emotion] += 1

    # Find the dominant emotion (if any)
    if emotions:
        dominant_emotion = max(emotions, key=emotions.get)
    else:
        dominant_emotion = "Neutral"

    # Display results
    print("\nSentence-wise Emotion Predictions:")
    for sentence in sentences:
        emotion = predict_emotion(sentence)
        if emotion != "Neutral":
            print(f"➡ {sentence}  →  {emotion}")

    print("\nOverall Emotion Count:", dict(emotions))
    print(f"\n🧠 Predicted Dominant Emotion: {dominant_emotion} 🎭")
