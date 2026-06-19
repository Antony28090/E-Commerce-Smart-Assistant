import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    lemmatizer = WordNetLemmatizer()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def preprocess_and_vectorize():
    print("Loading raw data...")
    raw_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'reviews.csv')
    df = pd.read_csv(raw_data_path)
    
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # We drop empty texts
    df = df[df['clean_text'].str.len() > 0]
    
    # Filter classes if there are too few samples in some classes for stratify
    value_counts = df['rating'].value_counts()
    valid_classes = value_counts[value_counts > 1].index
    df = df[df['rating'].isin(valid_classes)]
    
    # Stratified split based on rating
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df['rating'], 
        test_size=0.2, 
        stratify=df['rating'], 
        random_state=42
    )
    
    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Saving processed data and vectorizer...")
    with open(os.path.join(processed_dir, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train_vec, f)
    with open(os.path.join(processed_dir, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test_vec, f)
    with open(os.path.join(processed_dir, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(processed_dir, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(os.path.join(processed_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_and_vectorize()
