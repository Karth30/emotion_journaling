# Emotion Detection & Journaling Web App

## 📌 Overview
The **Emotion Detection & Journaling Web App** is an AI-powered platform that allows users to log their daily emotions and track their mood trends over time. It utilizes a **Bi-Directional LSTM + CNN model** to analyze user inputs and predict emotions, providing insightful visualizations to help users understand their emotional patterns.

## 🌐 Live Web App
Try out the Emotion Journaling Web App here: [Emotion Journaling App](https://emotion-detection-mlproject.streamlit.app/)

## 🚀 Features
- **Emotion Detection**: Uses a deep learning model to classify emotions from text.
- **Sentence-wise Emotion Analysis**: Breaks down journal entries sentence by sentence and assigns emotions.
- **Hard & Soft Emotion Categorization**: Provides both granular (sentence-level) and overall (soft) emotion scores.
- **Emotion Journaling**: Users can log their emotions daily.
- **Trend Visualization**: Interactive graphs to display emotion trends over time.
- **Modern UI**: Clean and intuitive interface using Streamlit.

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **Deep Learning**: Bi-Directional LSTM + CNN (TensorFlow/Keras)
- **Database**: Supabase (used for storing user journal entries and emotion trends)
- **Visualization**: Matplotlib, Seaborn

## ⚡ Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.10
- Virtual environment (optional but recommended)

### Steps to Run the Project
1. **Clone the repository**
   ```bash
   git clone https://github.com/kashika13/emotion_detection.git
   cd emotion_detection
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## 📊 Emotion Trend Visualization
- Each journal entry is analyzed sentence-wise to predict emotions.
- The **soft emotion** (overall emotion) is stored in the **Supabase database** for long-term trend analysis.
- Users can track their **emotional progress over time** using interactive graphs.

## 📝 Usage
1. **Enter your journal entry**, and the AI model will predict emotions for each sentence.
2. **View the overall soft emotion**, which is saved for tracking trends.
3. **Analyze emotion trends over time** with visualizations.

## 📂 Project Structure
```
├── app.py                  # Main Streamlit application
├── model/                  # Contains pre-trained ML and DL models
├── notebook/               # Contains jupyter notebook for differnt models and dataset
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
```

## 💡 Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.

## 📜 License
This project is licensed under the MIT License.



