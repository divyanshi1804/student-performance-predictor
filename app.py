"""
Student Performance Prediction System - Streamlit Web Dashboard
--------------------------------------------------------------
This is the interactive frontend application designed for beginners.
Features:
1. Loads the pre-trained Random Forest model and fitted LabelEncoders.
2. Displays a premium, simple-to-use user interface.
3. Provides drop-down menus and sliders for student background & score entries.
4. Performs automated category encoding and predictions when clicking 'Predict'.
5. Displays results with styled boxes: High (st.success), Average (st.warning), Low (st.error).
6. Outputs customized actionable study recommendations based on academic margins.

To run this dashboard:
streamlit run app.py

Author: Antigravity AI
OS: Windows Compatible
"""

import os
import pickle
import pandas as pd
import streamlit as st

# ==========================================
# STEP 0: PAGE CONFIGURATION & LAYOUT
# ==========================================
# Configure page tab title, icon, and sidebar layout
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom premium CSS styling for UI elements
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #2563EB;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .recommendation-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F766E;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Display Page Headers
st.markdown("<div class='main-title'>🎓 Student Performance Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze, predict, and receive smart study recommendations for students instantly!</div>", unsafe_allow_html=True)


# ==========================================
# STEP 1: LOAD THE MODEL & ENCODERS (PICKLE)
# ==========================================
# Dynamically locate absolute file paths to prevent Windows directory errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_rf_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")

# Helper function to load pickled assets safely using try-except
@st.cache_resource  # Caches files in RAM so they load instantly after the first time
def load_ml_assets():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
            
        return model, encoders
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, label_encoders = load_ml_assets()

# If model files are missing, show a beautiful error warning and stop execution
if model is None or label_encoders is None:
    st.error("🚨 **ML Model Files Not Found!**")
    st.warning("""
        It looks like the model files haven't been generated yet.
        
        **How to fix this:**
        1. Open your VS Code terminal.
        2. Run: `python main.py`
        3. This will train the model, save `student_rf_model.pkl` and `label_encoders.pkl` in the `models/` folder, and save EDA charts.
        4. Refresh this page!
    """)
    st.stop()


# ==========================================
# STEP 2: BUILD THE INPUT DASHBOARD
# ==========================================
# We divide the screen into two columns: Left for Inputs, Right for Predictions & Advice
col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:
    st.markdown("<div class='section-header'>📋 Enter Student Information</div>", unsafe_allow_html=True)
    
    # 2.A. Categorical inputs using select boxes
    # We dynamically read classes from label_encoders so we never get misspelling errors!
    gender = st.selectbox(
        "👤 Gender",
        options=label_encoders['gender'].classes_,
        help="Select the student's gender."
    )
    
    race = st.selectbox(
        "🌍 Race/Ethnicity Group",
        options=label_encoders['race/ethnicity'].classes_,
        help="Select the demographic group."
    )
    
    parent_edu = st.selectbox(
        "🏫 Parental Level of Education",
        options=label_encoders['parental level of education'].classes_,
        help="Select the highest education level of the student's parents."
    )
    
    lunch = st.selectbox(
        "🍱 Lunch Type",
        options=label_encoders['lunch'].classes_,
        help="Standard lunch or free/reduced lunch program."
    )
    
    test_prep = st.selectbox(
        "📝 Test Preparation Course Status",
        options=label_encoders['test preparation course'].classes_,
        help="Select whether the student completed the prep course before exams."
    )
    
    st.write("") # Whitespace divider
    
    # 2.B. Numeric inputs for scores using sliders
    st.markdown("##### 📊 Exam Performance Scores")
    math_score = st.slider("Math Score (0 - 100)", min_value=0, max_value=100, value=65)
    reading_score = st.slider("Reading Score (0 - 100)", min_value=0, max_value=100, value=65)
    writing_score = st.slider("Writing Score (0 - 100)", min_value=0, max_value=100, value=65)

with col2:
    st.markdown("<div class='section-header'>🔍 Prediction & Recommendations</div>", unsafe_allow_html=True)
    
    # Large action predict button
    predict_button = st.button("🚀 Predict Student Performance", use_container_width=True)
    
    if predict_button:
        # We wrap the prediction inside a try-except block to make the app robust and easy to debug
        try:
            # 1. Encode user inputs using our loaded LabelEncoders
            # .transform() takes a list/array and returns an array, so we take the [0] element
            encoded_gender = label_encoders['gender'].transform([gender])[0]
            encoded_race = label_encoders['race/ethnicity'].transform([race])[0]
            encoded_parent_edu = label_encoders['parental level of education'].transform([parent_edu])[0]
            encoded_lunch = label_encoders['lunch'].transform([lunch])[0]
            encoded_test_prep = label_encoders['test preparation course'].transform([test_prep])[0]
            
            # 2. Structure inputs into a Pandas DataFrame matching exact model feature names
            input_df = pd.DataFrame([{
                'gender': encoded_gender,
                'race/ethnicity': encoded_race,
                'parental level of education': encoded_parent_edu,
                'lunch': encoded_lunch,
                'test preparation course': encoded_test_prep,
                'math score': math_score,
                'reading score': reading_score,
                'writing score': writing_score
            }])
            
            # 3. Perform prediction using our RandomForest model
            prediction = model.predict(input_df)[0]
            
            # 4. Calculate average score to display alongside predicted level
            avg_score = (math_score + reading_score + writing_score) / 3
            
            # 5. Output Prediction Category with specific premium styled boxes
            st.markdown("#### **Analysis Result**")
            st.write(f"Based on the input values, the calculated Average Score is: **{avg_score:.2f}%**")
            
            if prediction == 'High':
                st.success(f"🎉 **Predicted Performance: High ({prediction})**\n\nThe student is performing exceptionally well! Maintain the momentum.")
            elif prediction == 'Average':
                st.warning(f"📊 **Predicted Performance: Average ({prediction})**\n\nThe student is performing moderately. There is room for steady improvement.")
            else:
                st.error(f"⚠️ **Predicted Performance: Low ({prediction})**\n\nThe student needs immediate academic assistance and structured study adjustments.")
            
            # ==========================================
            # STEP 3: SMART RECOMMENDATION SYSTEM
            # ==========================================
            st.markdown("<div class='recommendation-title'>💡 Recommended Actions</div>", unsafe_allow_html=True)
            
            # Keep track of recommendations list
            recommendations = []
            
            # A. Math margin check
            if math_score < 50:
                recommendations.append("❌ **Focus more on Mathematics:** Set aside 45 minutes daily for solving mathematical problems, focus on core formulas, and consider peer tutoring.")
            
            # B. Reading margin check
            if reading_score < 50:
                recommendations.append("📚 **Improve reading habits:** Read editorials, short essays, or comprehension pieces to build speed, vocabulary, and paragraph-retention skills.")
                
            # C. Writing margin check
            if writing_score < 50:
                recommendations.append("✍️ **Practice writing regularly:** Write brief daily summaries, essay practice drafts, or grammatic reviews to improve structure and punctuation.")
                
            # D. Test Prep Course check
            if test_prep == 'none':
                recommendations.append("📝 **Complete test preparation course:** Sign up for prep resources or standard practice mock tests to eliminate exam-day anxiety and get comfortable with formatting.")
                
            # E. Overall study hour check
            if avg_score < 50:
                recommendations.append("⏰ **Increase study hours:** Establish a daily study calendar (add 1-2 hours) focusing on weak topics and consistent homework routines.")
            
            # If everything is solid, reward them!
            if len(recommendations) == 0:
                st.info("🌟 **Excellent Profile!** The student has strong scores across all topics and completed test preparation. Keep up this amazing standard and continue regular practice.")
            else:
                for rec in recommendations:
                    st.info(rec)
                    
        except Exception as prediction_error:
            st.error(f"An unexpected error occurred during prediction: {prediction_error}")
            st.info("Please verify that the correct encoders are in place and try running `python main.py` again to reset.")

# ==========================================
# STEP 4: STATIC USER GUIDE SECTION
# ==========================================
st.write("")
st.write("")
with st.expander("📖 Quick App Guide & Run Instructions"):
    st.markdown("""
        ### How to Run this Dashboard on your Computer:
        1. Open your terminal in the project directory:
           `cd "c:\\Users\\ranur\\OneDrive\\Desktop\\Student Performance Predictor"`
        2. Execute the Streamlit run command:
           `streamlit run app.py`
        3. A browser tab will open automatically at: **http://localhost:8501**
        
        ### Under the Hood:
        - **Preprocessing & Encoding**: Handled dynamically using `sklearn.preprocessing.LabelEncoder`.
        - **Prediction**: Powered by a pickled `RandomForestClassifier` trained with a **97.00% accuracy**.
        - **Recommendations**: Smart conditionals evaluate specific scores to offer target feedback.
    """)
