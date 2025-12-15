T-Shirt Review Sentiment Classifier

This project is a Sentiment Analysis Web Application built using Natural Language Processing (NLP) and Machine Learning, deployed with Streamlit.
It classifies T-shirt reviews into Positive, Negative, or Neutral sentiments along with class probabilities.

The application uses a TF-IDF Vectorizer and a trained Machine Learning model to predict sentiment from user-entered text. 

app

Features

Clean and preprocess text using NLP techniques

TF-IDF based feature extraction

Machine Learning sentiment classification

Probability score for each sentiment class

Interactive and responsive Streamlit UI

Custom CSS styling for better user experience

Tech Stack

Python

Streamlit

Scikit-learn

NLTK

Pandas, NumPy

Joblib

Dependencies are listed in requirements.txt. 

requirements

Project Structure
├── app.py                  # Streamlit application
├── NLP_Task.csv            # Dataset used for training
├── NLP_Task.ipynb          # Model training notebook
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer
├── sentiment_model.pkl     # Trained ML model
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

How It Works

User enters a T-shirt review in the text area

Text is cleaned (lowercasing, stopword removal, punctuation removal)

Cleaned text is transformed using TF-IDF

ML model predicts:

Sentiment label

Probability for each class

Results are displayed visually in the web app

Installation & Setup
1. Clone the repository
git clone https://github.com/your-username/tshirt-sentiment-classifier.git
cd tshirt-sentiment-classifier

2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run app.py

Dataset

Dataset file: NLP_Task.csv

Contains T-shirt reviews with sentiment labels

Used for training and evaluation in NLP_Task.ipynb

Output Example

Input:
This t-shirt is amazing, great quality and perfect fit

Output:

Sentiment: Positive

Positive: 92%

Neutral: 6%

Negative: 2%

Future Improvements

Add deep learning models (LSTM / BERT)

Support multiple product categories

Deploy on cloud (AWS / Render / Hugging Face)

Add user authentication and review history

Author

Harsh Patel
Built using Streamlit, NLP, and Machine Learning
