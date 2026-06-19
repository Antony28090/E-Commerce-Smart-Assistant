import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_baseline():
    print("Loading processed data...")
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    
    with open(os.path.join(processed_dir, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(processed_dir, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(processed_dir, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(processed_dir, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
        
    print("Training Baseline Model (Logistic Regression)...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    print("Evaluating Baseline Model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    reports_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks') # Saving to notebooks for now
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, 'baseline_confusion_matrix.png'))
    print("Confusion matrix saved.")

if __name__ == "__main__":
    train_and_evaluate_baseline()
