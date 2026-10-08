"""
Student Performance Prediction System - Model Training & EDA Script
------------------------------------------------------------------

1. Load dataset using Pandas.
2. Preprocess data (handling duplicates, missing values, and column creation).
3. Encode categorical features using Scikit-Learn's LabelEncoder.
4. Perform Exploratory Data Analysis (EDA) and save visual plots.
5. Train a RandomForestClassifier to predict student performance levels.
6. Evaluate the model with Accuracy, Classification Report, and Confusion Matrix.
7. Save the trained model and fitted encoders using Pickle for the Streamlit web dashboard.

OS: Windows Compatible
"""
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import Scikit-Learn libraries for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# ==========================================
# STEP 0: SET UP DIRECTORIES AND PATHS
# ==========================================
print("=== Step 0: Setting up paths ===")
# Dynamically find the absolute path of this script to avoid Windows path issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Project Root Directory: {BASE_DIR}")
# Define relative paths for datasets, models, and screenshots
DATA_PATH = os.path.join(BASE_DIR, "dataset", "StudentsPerformance.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
# Programmatically create directories if they don't already exist to prevent errors
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
print("Directories initialized successfully.\n")
# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================
print("=== Step 1: Loading the Dataset ===")
# Read the CSV file into a Pandas DataFrame
try:
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded successfully!")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
except FileNotFoundError:
    print(f"Error: Could not find 'StudentsPerformance.csv' at {DATA_PATH}.")
    print("Please make sure the dataset is in the 'dataset' folder.")
    exit(1)
# Display the first 5 rows to understand the structure of the data
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("-" * 50 + "\n")
# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
print("=== Step 2: Data Preprocessing ===")
# A. Check for missing values in each column
print("Checking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("No missing values found! The dataset is clean.")
else:
    print(f"Found {missing_values.sum()} missing values. Filling them...")
    df.fillna(df.mean(numeric_only=True), inplace=True)
# B. Check and remove duplicate rows
print("\nChecking for duplicate rows:")
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows found: {duplicate_count}")
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed. New dataset shape: {df.shape}")
else:
    print("No duplicate rows to remove.")
print("-" * 50 + "\n")
# ==========================================
# STEP 3: FEATURE ENGINEERING
# ==========================================
print("=== Step 3: Feature Engineering ===")
# A. Create a new column 'average_score' by averaging math, reading, and writing scores
print("Calculating the 'average_score' column...")
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score']) / 3
print("Average scores calculated successfully.")
# B. Create the target column 'performance' based on the specified rules:
# High: >= 75 | Average: >= 50 | Low: < 50
print("Creating the target column 'performance' based on average scores...")
def categorize_performance(avg):
    if avg >= 75:
        return 'High'
    elif avg >= 50:
        return 'Average'
    else:
        return 'Low'
df['performance'] = df['average_score'].apply(categorize_performance)
# Show a breakdown of the new performance levels
print("\nStudent performance level breakdown:")
print(df['performance'].value_counts())
print("-" * 50 + "\n")
# ==========================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("=== Step 4: Exploratory Data Analysis (EDA) ===")
# Set visual style using Seaborn for premium aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
# 1. Countplot of performance levels
print("Generating Countplot of Performance Levels...")
plt.figure()
sns.countplot(x='performance', data=df, order=['High', 'Average', 'Low'], palette='viridis')
plt.title('Distribution of Student Performance Levels', fontsize=14, fontweight='bold')
plt.xlabel('Performance Level', fontsize=12)
plt.ylabel('Count of Students', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "performance_levels_distribution.png"), dpi=300)
plt.close()
# 2. Histogram of average scores
print("Generating Histogram of Average Scores...")
plt.figure()
sns.histplot(df['average_score'], bins=20, kde=True, color='purple')
plt.title('Distribution of Student Average Scores', fontsize=14, fontweight='bold')
plt.xlabel('Average Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "average_scores_histogram.png"), dpi=300)
plt.close()
# 3. Correlation Heatmap
print("Generating Correlation Heatmap...")
plt.figure()
# Calculate correlation matrix for numeric columns
corr_matrix = df[['math score', 'reading score', 'writing score', 'average_score']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Student Scores', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "scores_correlation_heatmap.png"), dpi=300)
plt.close()
# 4. Gender vs Average Score Graph
print("Generating Gender vs Average Score Graph...")
plt.figure()
sns.barplot(x='gender', y='average_score', data=df, palette='muted', ci=None)
plt.title('Gender vs Average Student Score', fontsize=14, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SCREENSHOTS_DIR, "gender_vs_average_score.png"), dpi=300)
plt.close()
print(f"All 4 EDA charts have been saved successfully inside: '{SCREENSHOTS_DIR}/'")
print("-" * 50 + "\n")
# ==========================================
# STEP 5: CATEGORICAL ENCODING
# ==========================================
print("=== Step 5: Encoding Categorical Columns ===")
# Columns that contain text and need to be encoded to numbers for the ML model
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
# Create a dictionary to hold the label encoders so we can save and reuse them in Streamlit
label_encoders = {}
print("Encoding categorical columns using LabelEncoder:")
for col in categorical_columns:
    le = LabelEncoder()
    # Fit the encoder to the column data and transform it to numeric values
    df[col] = le.fit_transform(df[col].astype(str))
    # Store the fitted encoder in our dictionary
    label_encoders[col] = le
    print(f" - Encoded '{col}' successfully. Classes: {list(le.classes_)}")
print("-" * 50 + "\n")
# ==========================================
# STEP 6: DATA SPLITTING (TRAIN/TEST SPLIT)
# ==========================================
print("=== Step 6: Train/Test Split ===")
# Features (Inputs for our ML Model)
# We train the model on student demographics AND their individual exam scores
feature_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score', 'writing score']
X = df[feature_cols]
# Target Variable (What we want to predict)
y = df['performance']
# Split the dataset into 80% Training data and 20% Testing data
# We set random_state=42 to ensure our results are reproducible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total training samples: {X_train.shape[0]}")
print(f"Total testing samples: {X_test.shape[0]}")
print("-" * 50 + "\n")
# ==========================================
# STEP 7: MODEL TRAINING (RANDOM FOREST)
# ==========================================
print("=== Step 7: Training the RandomForestClassifier ===")
# Initialize the Random Forest Classifier
# We set random_state=42 for reproducibility as required
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train (fit) the model using our training data
model.fit(X_train, y_train)
print("Model training completed successfully!")
print("-" * 50 + "\n")
# ==========================================
# STEP 8: MODEL EVALUATION
# ==========================================
print("=== Step 8: Evaluating the Model ===")
# Predict the student performances on the unseen test set
y_pred = model.predict(X_test)
# A. Calculate and print Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Test Accuracy: {accuracy * 100:.2f}%")
# B. Print Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# C. Calculate and print Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['High', 'Average', 'Low'])
print(cm)
print("-" * 50 + "\n")
# ==========================================
# STEP 9: SAVE TRAINED MODEL AND ENCODERS
# ==========================================
print("=== Step 9: Saving Model and Encoders ===")
# Paths for saving the pickle files
model_file_path = os.path.join(MODELS_DIR, "student_rf_model.pkl")
encoders_file_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
# Save the trained RandomForest model
with open(model_file_path, 'wb') as f:
    pickle.dump(model, f)
print(f"1. Trained model saved to: {model_file_path}")
# Save the fitted label encoders dictionary
with open(encoders_file_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"2. Label encoders saved to: {encoders_file_path}")
print("\n=== Pipeline Executed Successfully! Project files are ready for Streamlit dashboard. ===")
