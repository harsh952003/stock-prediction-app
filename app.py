import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# First time download
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# ------------ Load model and vectorizer -------------
@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("sentiment_model.pkl")
    return tfidf, model

tfidf, model = load_artifacts()

# ------------ Text cleaning (same as training) -------------
def clean_text(text, remove_stopwords=True):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words]
        text = " ".join(tokens)

    return text

def predict_sentiment(text):
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]  # probability for each class
    # Mapping class -> probability
    class_probs = dict(zip(model.classes_, probs))
    return pred, class_probs

# ---------------- CUSTOM CSS STYLING ---------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .title-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .title-container h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border-left: 5px solid #667eea;
    }
    
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        margin: 2rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #155724;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 5px 15px rgba(132, 250, 176, 0.3);
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #721c24;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 5px 15px rgba(250, 112, 154, 0.3);
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 5px 15px rgba(168, 237, 234, 0.3);
    }
    
    .probability-card {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    
    .probability-card:hover {
        transform: translateX(5px);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        border: none;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1.4rem;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    h2 {
        color: #667eea;
        font-weight: 600;
    }
    
    .emoji {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- STREAMLIT UI ---------------------

# Header with gradient
st.markdown("""
    <div class="title-container">
        <h1>👕 T-shirt Review Sentiment Classifier</h1>
        <p style="font-size: 1.1rem; margin-top: 0.5rem; opacity: 0.95;">NLP & Machine Learning Powered Analysis</p>
    </div>
""", unsafe_allow_html=True)

# Info box
# st.markdown("""
#     <div class="info-box">
#         <p style="font-size: 1.05rem; margin: 0; line-height: 1.6;">
#             <strong>📊 About:</strong> Yeh website tumhare <strong>NLP_Task.csv</strong> pe trained machine learning model use karti hai 
#             jo review ko <strong>Positive / Negative / Neutral</strong> classify karti hai.
#         </p>
#     </div>
# """, unsafe_allow_html=True)

# Input section
# st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown("### ✍️ Enter Your T-shirt Review")
review_text = st.text_area(
    "Review Text",
    placeholder="Type your review here... (e.g., 'This t-shirt is amazing! Great quality and perfect fit.')",
    height=150,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 Predict Sentiment", use_container_width=True)

if predict_btn:
    if review_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔄 Analyzing sentiment..."):
            label, probs = predict_sentiment(review_text)
            label_cap = label.capitalize()
        
        # st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Analysis Results")
        
        # Sentiment display
        if label.lower() == "positive":
            st.markdown(f"""
                <div class="sentiment-positive">
                    <span class="emoji">😊</span> Sentiment: {label_cap}
                </div>
            """, unsafe_allow_html=True)
        elif label.lower() == "negative":
            st.markdown(f"""
                <div class="sentiment-negative">
                    <span class="emoji">😞</span> Sentiment: {label_cap}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="sentiment-neutral">
                    <span class="emoji">😐</span> Sentiment: {label_cap}
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Class Probabilities")
        
        # Probability bars with styling
        for cls, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
            emoji = emoji_map.get(cls.lower(), "📊")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                    <div class="probability-card">
                        <strong>{emoji} {cls.capitalize()}</strong>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div style="text-align: right; padding-top: 1rem; font-weight: 600; color: #667eea;">
                        {p*100:.1f}%
                    </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(p, text=f"{p*100:.2f}%")
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Made with Harsh Patel using Streamlit & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)