# Cell 1: Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, RocCurveDisplay, precision_recall_curve)  # ADDED precision_recall_curve

print("All libraries imported successfully!")

# Load the dataset
dataset_path = 'heart-attack-risk-prediction-dataset.csv'
df = pd.read_csv(dataset_path)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Cell 3: Exploratory Data Analysis (EDA)
print("\n=== DATA INFO ===")
df.info()

print("\n=== STATISTICAL SUMMARY ===")
print(df.describe())

print("\n=== TARGET VARIABLE DISTRIBUTION ===")
risk_counts = df['Heart Attack Risk'].value_counts()
print(risk_counts)

plt.figure(figsize=(8, 5))
sns.countplot(x='Heart Attack Risk', data=df)
plt.title('Distribution of Heart Attack Risk')
plt.tight_layout()
plt.show()

# Cell 4: Data Preprocessing & Cleaning
df_clean = df.copy()

# Handle missing values
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Create individual label encoders for each categorical column
label_encoders = {}
categorical_cols = ['Gender', 'Smoking', 'Obesity', 'Diabetes', 'Previous Heart Problems', 'Medication Use']

for col in categorical_cols:
    if col in df_clean.columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

print("Data after encoding categorical text to numbers:")
print(df_clean.head())

# Cell 5: Split the Data into Features (X) and Target (y)
X = df_clean.drop('Heart Attack Risk', axis=1)
y = df_clean['Heart Attack Risk']

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Cell 6: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set - Features:", X_train.shape, "Target:", y_train.shape)
print("Testing set  - Features:", X_test.shape, "Target:", y_test.shape)

# Cell 7: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete.")

# Cell 8: Train Multiple Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5)
}

model_scores = {}
model_predictions = {}
model_probabilities = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Get prediction probabilities if available
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = None
    
    accuracy = accuracy_score(y_test, y_pred)
    model_scores[name] = accuracy
    model_predictions[name] = y_pred
    model_probabilities[name] = y_pred_proba
    
    print(f"{name} Accuracy: {accuracy:.4f}")

# Compare model performance
print("\n=== MODEL COMPARISON ===")
for name, score in model_scores.items():
    print(f"{name}: {score:.4f}")

# Cell 9: Model Accuracy Comparison Visualization
plt.figure(figsize=(10, 6))
models_list = list(model_scores.keys())
scores_list = list(model_scores.values())

bars = plt.bar(models_list, scores_list, color=['#FF9999', '#66B2FF', '#99FF99'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add value labels on top of each bar
for bar, score in zip(bars, scores_list):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Cell 10: Confusion Matrices for Each Model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, y_pred) in enumerate(model_predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name} Confusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Cell 11: ROC Curves for Each Model
plt.figure(figsize=(10, 8))

for name, model in models.items():
    if model_probabilities[name] is not None:
        try:
            RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, name=name)
        except ValueError as e:
            print(f"Error plotting ROC curve for {name}: {e}")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curves')
plt.legend()
plt.show()

# Cell 12: Feature Importance for Tree-Based Models
tree_models = {k: v for k, v in models.items() if hasattr(v, 'feature_importances_')}

if tree_models:
    fig, axes = plt.subplots(1, len(tree_models), figsize=(6*len(tree_models), 5))
    
    if len(tree_models) == 1:
        axes = [axes]  # Make it iterable if only one model
    
    for idx, (name, model) in enumerate(tree_models.items()):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        axes[idx].barh(feature_importance['feature'][:7], feature_importance['importance'][:7], color='skyblue')
        axes[idx].set_title(f'Feature Importance - {name}')
        axes[idx].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.show()
else:
    print("No tree-based models available for feature importance visualization")

# Cell 13: Decision Tree Visualization
if "Decision Tree" in models:
    plt.figure(figsize=(20, 10))
    plot_tree(models["Decision Tree"], 
              feature_names=X.columns, 
              class_names=['Low Risk', 'High Risk'], 
              filled=True, 
              rounded=True, 
              proportion=True,
              max_depth=3)  # Show only first 3 levels for clarity
    plt.title("Decision Tree (First 3 Levels)")
    plt.show()

# Cell 14: Classification Reports
for name, y_pred in model_predictions.items():
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(y_test, y_pred))

# Cell 15: Precision-Recall Curves
plt.figure(figsize=(10, 8))

for name, model in models.items():
    if model_probabilities[name] is not None:
        try:
            precision, recall, _ = precision_recall_curve(y_test, model_probabilities[name])
            plt.plot(recall, precision, label=name, linewidth=2)
        except Exception as e:
            print(f"Error plotting Precision-Recall curve for {name}: {e}")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)
plt.show()

# Cell 16: Detailed Analysis of Best Model
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]
y_pred = model_predictions[best_model_name]

print(f"\n=== DETAILED ANALYSIS OF BEST MODEL: {best_model_name} ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if model_probabilities[best_model_name] is not None:
    try:
        roc_auc = roc_auc_score(y_test, model_probabilities[best_model_name])
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    except:
        print("Could not calculate ROC-AUC score")

# Feature importance for the best model (if applicable)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()

# ======== Save Models for API ========
print("\n=== SAVING MODELS FOR API ===")

if not os.path.exists('model'):
    os.makedirs('model')
    print("Created 'model' directory")

# Save all three models
joblib.dump(models["Logistic Regression"], 'model/log_reg_model.pkl')
print("Logistic Regression model saved to model/log_reg_model.pkl")

joblib.dump(models["Random Forest"], 'model/rf_model.pkl')
print("Random Forest model saved to model/rf_model.pkl")

joblib.dump(models["Decision Tree"], 'model/decision_tree_model.pkl')
print("Decision Tree model saved to model/decision_tree_model.pkl")

joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to model/scaler.pkl")

joblib.dump(label_encoders, 'model/label_encoders.pkl')
print("Label encoders saved to model/label_encoders.pkl")

feature_names = X.columns.tolist()
joblib.dump(feature_names, 'model/feature_names.pkl')
print("Feature names saved to model/feature_names.pkl")

print("\nML Training Complete!")
print("Results Summary:")
print(f"  Logistic Regression Accuracy: {model_scores['Logistic Regression']:.4f}")
print(f"  Random Forest Accuracy: {model_scores['Random Forest']:.4f}")
print(f"  Decision Tree Accuracy: {model_scores['Decision Tree']:.4f}")
print(f"\nBest Model: {best_model_name} with accuracy: {model_scores[best_model_name]:.4f}")
print("\nReady for API deployment!")
print("  Run: python api.py")