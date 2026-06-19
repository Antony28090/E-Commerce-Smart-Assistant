import streamlit as st
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up NLTK (downloads if missing)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Page Configuration
st.set_page_config(page_title="E-Commerce Smart Assistant", layout="centered")

# --- Helper Functions ---
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    
    with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(processed_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    lemmatizer = WordNetLemmatizer()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# --- UI Layout ---
st.title("E-Commerce Smart Assistant")
st.markdown("Analyze product reviews using our NLP classification engine. Enter a review below to predict its sentiment and rating.")

model, vectorizer = load_models()

user_input = st.text_area("Enter a product review:", height=150, placeholder="E.g., I bought this product and it was absolutely fantastic! Highly recommend...")

if st.button("Analyze Review", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # 1. Preprocess
            cleaned_text = clean_text(user_input)
            
            if cleaned_text.strip() == "":
                 st.error("The review didn't contain enough meaningful words to analyze after removing stopwords.")
            else:
                # 2. Vectorize
                vec_input = vectorizer.transform([cleaned_text])
                
                # 3. Predict Rating
                prediction = model.predict(vec_input)[0]
                
                # Try to get prediction probabilities if the model supports it
                try:
                    probs = model.predict_proba(vec_input)[0]
                    confidence = max(probs) * 100
                except AttributeError:
                    confidence = 100.0 # Fallback if model doesn't support predict_proba
                    
                st.subheader("Analysis Results")
                
                # Visual Metric Cards
                col1, col2 = st.columns(2)
                
                with col1:
                    # Map rating to sentiment
                    if prediction >= 4:
                        sentiment = "Positive"
                    elif prediction == 3:
                        sentiment = "Neutral"
                    else:
                        sentiment = "Negative"
                        
                    st.metric("Predicted Sentiment", sentiment)
                    
                with col2:
                    st.metric("Predicted Rating", f"{prediction} / 5 Stars")
                    
                st.progress(confidence / 100)
                st.caption(f"Model Confidence: {confidence:.2f}%")
                
                st.markdown("---")
                st.write("**Extracted Actionable Tags (Demo Category):** `Electronics`, `General`")
