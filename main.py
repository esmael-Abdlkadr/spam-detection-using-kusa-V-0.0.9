import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from kusa.client import SecureDatasetClient
from sklearn.metrics import precision_recall_curve, average_precision_score,f1_score
from sklearn.preprocessing import label_binarize
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB # Good for TF-IDF features
from sklearn.tree import DecisionTreeClassifier

# 🔧 Framework flag: sklearn | tensorflow | pytorch
TRAINING_FRAMEWORK = "sklearn"
TARGET_COLUMN = "Category"

# --- CHOOSE YOUR SKLEARN MODEL HERE ---
# Comment out/uncomment to select a model
# SELECTED_SKLEARN_MODEL = RandomForestClassifier
# SELECTED_SKLEARN_MODEL = GradientBoostingClassifier
SELECTED_SKLEARN_MODEL = LogisticRegression
# SELECTED_SKLEARN_MODEL = SVC
# SELECTED_SKLEARN_MODEL = KNeighborsClassifier
# SELECTED_SKLEARN_MODEL = MultinomialNB
# SELECTED_SKLEARN_MODEL = DecisionTreeClassifier
#SELECTED_SKLEARN_MODEL = AdaBoostClassifier

load_dotenv(override=True)

# ✅ Framework-aware training factory
def train_model_factory(framework="sklearn", model_class=None, fixed_params=None):
    fixed_params = fixed_params or {}
    if framework == "sklearn":
        def train_model(X, y, **params):
            sig = signature(model_class.__init__)
            accepted = set(sig.parameters.keys())
            valid = {k: v for k, v in {**fixed_params, **params}.items() if k in accepted}
            return model_class(**valid).fit(X, y)
        return train_model

    else:
        raise ValueError("Unsupported framework selected")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={avg_precision:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


def plot_threshold_analysis(y_true, y_proba, title="Threshold Analysis"):
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p, r = precision_recall_curve(y_true, preds)[:2]
        precisions.append(p[1])
        recalls.append(r[1])
        f1s.append(f1_score(y_true, preds))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="green")
    plt.plot(thresholds, f1s, label="F1 Score", color="red")
    plt.axvline(x=0.5, linestyle='--', color='gray', label="Threshold = 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def main():
    PUBLIC_ID = os.getenv("PUBLIC_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    initialization = client.initialize()

    getPreviewOfDataset = client.preview()
    print(getPreviewOfDataset)
    client.fetch_and_decrypt_batch(batch_size=500)

    print("⚙️ Configuring preprocessing...")
    client.configure_preprocessing({ 
        "tokenizer": "nltk",
        "stopwords": True,
        "reduction": "pca",  # Optional for NLP; skip or change for tabular tfidf,pca or tfidf_pca
        "target_column": TARGET_COLUMN,
        "target_encoding": "auto"  # or dict, or "none" dict means like  {"ham": 0, "spam": 1}

    })
    client.run_preprocessing()
  
    
    train_model = train_model_factory(TRAINING_FRAMEWORK, model_class=SELECTED_SKLEARN_MODEL)
    # --- Set Model-Specific Hyperparameters ---
    if SELECTED_SKLEARN_MODEL == RandomForestClassifier:
        hyperparams = {"n_estimators": 100, "class_weight": "balanced", "random_state": 42}
    elif SELECTED_SKLEARN_MODEL == GradientBoostingClassifier:
        hyperparams = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42}
    elif SELECTED_SKLEARN_MODEL == LogisticRegression:
        hyperparams = {"solver": "liblinear", "class_weight": "balanced", "random_state": 42, "C": 1.0}
    elif SELECTED_SKLEARN_MODEL == SVC:
        # SVC can be slow on large datasets. Ensure 'probability=True' for predict_proba.
        hyperparams = {"kernel": "rbf", "class_weight": "balanced", "probability": True, "random_state": 42, "C": 1.0}
    elif SELECTED_SKLEARN_MODEL == KNeighborsClassifier:
        hyperparams = {"n_neighbors": 5, "weights": "distance"}
    elif SELECTED_SKLEARN_MODEL == MultinomialNB:
        hyperparams = {"alpha": 1.0} # Smoothing parameter
    elif SELECTED_SKLEARN_MODEL == DecisionTreeClassifier:
        hyperparams = {"max_depth": 5, "class_weight": "balanced", "random_state": 42}
    elif SELECTED_SKLEARN_MODEL == AdaBoostClassifier:
        hyperparams = {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42}
           
    print("🚀 Training model...")
    client.train(
         user_train_func=train_model, 
         hyperparams=hyperparams, target_column=TARGET_COLUMN,
         task_type="classification", framework=TRAINING_FRAMEWORK, 
    )

    print("📈 Evaluating model...")
    results = client.evaluate()
    print("\n✅ Evaluation Accuracy:", results["accuracy"])
    print("📊 Classification Report:\n", results["report"])

    print("📉 Visualizing confusion matrix...")
    
    y_true = client._SecureDatasetClient__y_val
    y_pred = client.predict(client._SecureDatasetClient__X_val)
    
    plot_confusion_matrix(y_true, y_pred)
            
    # 🔥 Precision-Recall Curve
    if TRAINING_FRAMEWORK == "sklearn":
        y_proba = client._SecureDatasetClient__trained_model.predict_proba(client._SecureDatasetClient__X_val)[:, 1]
    elif TRAINING_FRAMEWORK == "tensorflow":
        y_proba = client._SecureDatasetClient__trained_model.predict(client._SecureDatasetClient__X_val).flatten()
    elif TRAINING_FRAMEWORK == "pytorch":
        model = client._SecureDatasetClient__trained_model # Problematic direct access
        model.eval()
        with torch.no_grad():
            # This is where the error happens:
            inputs = torch.tensor(client._SecureDatasetClient__X_val.values, dtype=torch.float32)
            y_proba = model(inputs).numpy().flatten()
            
    plot_precision_recall_curve(y_true, y_proba)

    # Show thresholds
    plot_threshold_analysis(y_true, y_proba)

    print("💾 Saving trained model...")
    
    client.save_model(f"secure_model_{TRAINING_FRAMEWORK}.model")

    print("\n✅ Done! 🎉")

    # print("\n🧪 Preview:\n", initialization.get("preview").head())

if __name__ == "__main__":
    main()
