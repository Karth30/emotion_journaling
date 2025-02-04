import gensim
import joblib
from nltk.corpus import stopwords
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np

lemmatizer=WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 

# Load Word2Vec Model
#word2vec_model = gensim.models.Word2Vec.load("model/word2vec_model.bin")
# Loading the model
word2vec_model = joblib.load('model/word2vec_model.joblib')

# Load Trained ML Model
#with open("model/emotion_classifier_logistic.pkl", "rb") as file:
    #loaded_model = pickle.load(file)
loaded_model=joblib.load('model/emotion_classifier_logistic.joblib')

# Preprocessing Function
def preprocess_text(text):
    words = []
    sent_token = sent_tokenize(text)  # Sentence tokenization
    for sent in sent_token:
        tokens = simple_preprocess(sent)  # Word tokenization & preprocessing
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords and lemmatize
        words.extend(filtered_tokens)
    return words

# Define Function to Get Sentence Vector
def avg_word2vec(doc):
    vectors = [word2vec_model.wv[word] for word in doc if word in word2vec_model.wv.index_to_key]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

def predict_emotion(text):
    text_preprocessed = preprocess_text(text)
    text_vector = np.array([avg_word2vec(text_preprocessed)])  
    predicted_emotion = loaded_model.predict(text_vector)[0] 
    return predicted_emotion


# Test Example
if __name__ == "__main__":
    test_sentence=input("Enter sentence: ")
    print(f"Predicted Emotion: {predict_emotion(test_sentence)}")
