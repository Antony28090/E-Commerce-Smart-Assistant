# E-Commerce Smart Assistant 🛍️

This repository contains the end-to-end 4-week internship project: an **E-Commerce Smart Assistant**. The assistant extracts product reviews, performs data preprocessing and feature engineering, trains machine learning models to classify sentiments/ratings, and deploys the results via a Streamlit web application.

## Project Structure

```
├── app.py                     # Streamlit web application (Deployment)
├── requirements.txt           # Project dependencies
├── data/
│   ├── raw/                   # Raw mined data (CSV files)
│   └── processed/             # Cleaned text, TF-IDF vectorizers, and train/test splits
├── models/                    # Saved machine learning models (e.g., best_model.pkl)
├── notebooks/                 # Jupyter Notebooks for EDA and analysis
└── src/
    ├── data/                  # Scripts for downloading datasets and web scraping (scraper.py)
    ├── features/              # Text preprocessing and vectorization pipelines
    └── models/                # Training scripts (Baseline vs Advanced)
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "Global Logic"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Data & Preprocess:**
   ```bash
   # Download the raw Amazon Electronics reviews dataset
   python src/data/download_dataset.py
   
   # Alternatively, test the BeautifulSoup web scraper on books.toscrape.com
   python src/data/scraper.py
   
   # Preprocess the text and create TF-IDF features
   python src/features/preprocess.py
   ```

4. **Train Models:**
   ```bash
   # Train the Logistic Regression Baseline
   python src/models/train_baseline.py
   
   # Train the Tuned Logistic Regression Advanced Model with GridSearchCV
   python src/models/train_advanced.py
   ```

## Running the Application

To start the Streamlit web interface and interact with the AI assistant:
```bash
streamlit run app.py
```
This will open the web application in your default browser where you can paste any product review and receive real-time sentiment predictions!

## Features
- **Data Mining**: Includes both direct data loading from academic datasets and a custom BeautifulSoup web scraper with pagination and simulated rate-limiting.
- **NLP Text Cleansing**: Removes HTML, punctuation, applies lowercasing, stop-word removal, and lemmatization using NLTK.
- **Machine Learning**: Compares Logistic Regression against Random Forest via GridSearchCV hyperparameter tuning.
- **Interactive UI**: Real-time evaluation of unseen text reviews using Streamlit.
