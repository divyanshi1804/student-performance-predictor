# 🎓 Student Performance Prediction System

A complete, beginner-friendly Machine Learning project in Python designed to analyze student demographic backgrounds and test scores, predict their overall performance tier (**High**, **Average**, or **Low**), and provide actionable, custom-tailored academic recommendations through an interactive web dashboard.

This project is built using simple Python, standard data-science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`), and the `Streamlit` dashboarding library.

---

## 📁 Project Folder Structure

```text
Student-Performance-Prediction/
│
├── dataset/
│   └── StudentsPerformance.csv     # The Kaggle CSV dataset
│
├── models/
│   ├── student_rf_model.pkl        # Pickle file containing the trained RandomForestClassifier
│   └── label_encoders.pkl          # Pickle file containing fitted LabelEncoder objects
│
├── screenshots/
│   ├── performance_levels_distribution.png  # Countplot of performance tiers
│   ├── average_scores_histogram.png        # Score distribution histogram
│   ├── scores_correlation_heatmap.png       # Correlation heatmap of exam scores
│   └── gender_vs_average_score.png         # Gender vs average score comparison
│
├── report/
│   └── project_report.md           # Simple and structured academic report outline
│
├── ppt/
│   └── presentation_outline.md     # slide-by-slide PPT script outline
│
├── main.py                         # EDA, preprocessing, model training & saving script
├── app.py                          # Streamlit web dashboard & recommendation engine
├── requirements.txt                # Project dependency list
└── README.md                       # Complete project instructions (This file)
```

---

## 🛠️ Step-by-Step Installation & Run Guide

To run this project on a Windows computer using VS Code, follow these steps:

### Step 1: Open VS Code & Open Terminal
Open VS Code, select **File > Open Folder**, and open the `Student Performance Predictor` folder. Press **Ctrl + `** (or go to **Terminal > New Terminal**) to open the command line terminal.

### Step 2: Install Libraries
Ensure you have all the necessary libraries installed by executing:
```powershell
pip install -r requirements.txt
```

### Step 3: Run Model Training & EDA
Before launching the web app, run `main.py` to process the data, create EDA visualization plots, train the machine learning classifier, and save the binary files:
```powershell
python main.py
```
* **What this does:**
  * Checks for duplicates and missing values.
  * Creates an `average_score` column and a target `performance` label.
  * Saves 4 data visualization graphs in the `screenshots/` directory.
  * Encodes text categories to numbers.
  * Trains a `RandomForestClassifier` (achieving **97.00% accuracy**!).
  * Pickles the trained model and label encoders in the `models/` directory.

### Step 4: Launch the Web Dashboard
Now, start the Streamlit interactive dashboard by running:
```powershell
streamlit run app.py
```
* **What this does:**
  * Instantly opens a local browser tab at **http://localhost:8501**.
  * Displays dropdowns for student details and sliders for exam scores.
  * Predicts performance upon clicking "Predict" and outputs clear study recommendations.

---

## 🧠 Machine Learning Details (Beginner Friendly)

### 1. The Dataset
The Kaggle `StudentsPerformance.csv` dataset contains 1,000 records of students' demographic data and three exam scores:
* **Categorical columns:** `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`.
* **Numerical columns:** `math score`, `reading score`, `writing score`.

### 2. Label Encoding
Machine learning algorithms only understand numbers. We use Scikit-Learn's `LabelEncoder` to convert categories into integers (e.g., `'female'` becomes `0` and `'male'` becomes `1`). These encoders are saved as a dictionary using `pickle` so that the Streamlit app uses the exact same translations for new user inputs.

### 3. Classification with Random Forest
We train a **RandomForestClassifier** with `random_state=42`. 
* **How it works:** Think of a Random Forest as an ensemble of many simple "decision trees" (like a flow chart). Each tree votes on the final performance category of a student based on their demographic background and exam scores. The category with the most votes wins!
* **Target Categories:**
  * 🥇 **High:** Average score $\ge 75$
  * 🥈 **Average:** Average score $\ge 50$
  * 🥉 **Low:** Average score $< 50$

---

## 💡 Recommendation System
The system doesn't just predict performance; it calculates academic weak points and outputs specific actionable feedback:
* **Math Score < 50** $\rightarrow$ "Focus more on Mathematics"
* **Reading Score < 50** $\rightarrow$ "Improve reading habits"
* **Writing Score < 50** $\rightarrow$ "Practice writing regularly"
* **No test preparation** $\rightarrow$ "Complete test preparation course"
* **Overall average score < 50** $\rightarrow$ "Increase study hours"
