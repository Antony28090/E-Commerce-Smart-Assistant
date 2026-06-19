import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_advanced():
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
        
    print("Training Advanced Model (Tuned Logistic Regression) with GridSearchCV...")
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    
    param_grid = {
        'C': [0.1, 1.0, 10.0]
    }
    
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    print("Evaluating Advanced Model...")
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Advanced Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    reports_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'notebooks')
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, 'advanced_confusion_matrix.png'))
    print("Confusion matrix saved.")
    
    # Save the best model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {best_model_path}")

if __name__ == "__main__":
    train_and_evaluate_advanced()
