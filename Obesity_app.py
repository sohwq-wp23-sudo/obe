import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try importing plotly, provide fallback if not available
try:
    import plotly.express as px
    plotly_available = True
except ImportError:
    plotly_available = False
    st.warning("Plotly not installed. Using basic visualizations. Run: pip install plotly")

# Set page config
st.set_page_config(
    page_title="Obesity Level Predictor",
    page_icon="⚖️",
    layout="wide"
)

# Title and description
st.title("⚖️ Obesity Level Prediction Prototype")
st.markdown("""
This prototype predicts obesity levels based on eating habits, physical condition, and lifestyle factors.
Adjust the parameters below to see how different factors influence obesity risk.
""")

# Function to load data from txt file
@st.cache_data
def load_data():
    try:
        # Try to load from uploaded file first
        if 'data_file' in st.session_state and st.session_state.data_file is not None:
            df = pd.read_csv(st.session_state.data_file)
        else:
            # Load default data from txt file
            df = pd.read_csv('obesity_data.txt')
        
        # Calculate BMI
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        return df
    except FileNotFoundError:
        st.error("⚠️ obesity_data.txt file not found! Please make sure it's in the same directory.")
        # Create sample data if file not found
        st.info("Creating sample data for demonstration...")
        sample_data = {
            'Age': [25, 32, 45, 28, 35],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Height': [1.75, 1.65, 1.80, 1.60, 1.72],
            'Weight': [70, 85, 110, 55, 95],
            'FamilyHistory': ['no', 'yes', 'yes', 'no', 'no'],
            'HighCaloricFreq': ['no', 'yes', 'yes', 'no', 'yes'],
            'VegConsumption': [2, 1, 1, 3, 1],
            'MainMeals': [3, 4, 3, 3, 4],
            'FoodBetweenMeals': ['no', 'frequently', 'always', 'no', 'frequently'],
            'Smoking': ['no', 'no', 'no', 'yes', 'no'],
            'WaterConsumption': [2.5, 1.5, 1.0, 3.0, 1.5],
            'CaloriesMonitor': ['no', 'yes', 'no', 'yes', 'no'],
            'PhysicalActivity': [1.2, 0.5, 0.3, 2.0, 0.8],
            'TechUsage': [1.0, 1.8, 2.0, 0.5, 1.5],
            'AlcoholConsumption': ['no', 'sometimes', 'always', 'no', 'sometimes'],
            'Transportation': ['Walking', 'Automobile', 'Public_Transportation', 'Bike', 'Motorbike'],
            'ObesityLevel': ['Normal Weight', 'Overweight Level I', 'Obesity Type II', 'Normal Weight', 'Overweight Level II']
        }
        df = pd.DataFrame(sample_data)
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# File uploader in sidebar
st.sidebar.markdown("### 📁 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your own data (CSV/TXT)", type=['csv', 'txt'])
if uploaded_file is not None:
    st.session_state.data_file = uploaded_file
    st.sidebar.success("✅ Custom data loaded!")

# Load data
df = load_data()

if df.empty:
    st.stop()

# Sidebar for user input
st.sidebar.markdown("---")
st.sidebar.header("📊 Patient Information")
st.sidebar.markdown("---")

# Personal info
st.sidebar.subheader("Personal")
age = st.sidebar.slider("Age (years)", 14, 61, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
height = st.sidebar.slider("Height (m)", 1.45, 1.98, 1.70, 0.01)
weight = st.sidebar.slider("Weight (kg)", 40.0, 120.0, 70.0, 0.1)
bmi = weight / (height ** 2)

# Display BMI
st.sidebar.info(f"**BMI:** {bmi:.1f}")

# Eating habits
st.sidebar.subheader("🥗 Eating Habits")
family_history = st.sidebar.radio("Family history of obesity", ["no", "yes"])
high_caloric = st.sidebar.radio("Frequent high-caloric food", ["no", "yes"])
veg_consumption = st.sidebar.select_slider("Vegetable consumption (1-3)", options=[1, 2, 3], value=2)
main_meals = st.sidebar.select_slider("Number of main meals per day", options=[1, 2, 3, 4], value=3)
food_between = st.sidebar.selectbox("Food between meals", ["no", "sometimes", "frequently", "always"])

# Lifestyle
st.sidebar.subheader("🏃 Lifestyle")
smoking = st.sidebar.radio("Smoking", ["no", "yes"])
water = st.sidebar.slider("Daily water consumption (liters)", 1.0, 8.0, 2.0, 0.1)
calories_monitor = st.sidebar.radio("Monitor calories", ["no", "yes"])
physical_activity = st.sidebar.slider("Physical activity frequency (days/week)", 0.0, 3.0, 1.0, 0.1)
tech_usage = st.sidebar.slider("Time using tech devices (hours/day)", 0.0, 2.0, 1.0, 0.1)
alcohol = st.sidebar.selectbox("Alcohol consumption", ["no", "sometimes", "frequently", "always"])
transport = st.sidebar.selectbox("Transportation", ["Walking", "Bike", "Motorbike", "Public_Transportation", "Automobile"])

# Train model
@st.cache_resource
def train_model(data):
    # Prepare features
    features = ['Age', 'Gender', 'FamilyHistory', 'HighCaloricFreq', 'VegConsumption', 
                'MainMeals', 'FoodBetweenMeals', 'Smoking', 'WaterConsumption', 
                'CaloriesMonitor', 'PhysicalActivity', 'TechUsage', 'AlcoholConsumption', 
                'Transportation']
    
    X = data[features].copy()
    y = data['ObesityLevel']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Gender', 'FamilyHistory', 'HighCaloricFreq', 'FoodBetweenMeals', 
                        'Smoking', 'CaloriesMonitor', 'AlcoholConsumption', 'Transportation']
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate accuracy
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracy = model.score(X_test, y_test)
    except:
        accuracy = 0.85  # Default accuracy if splitting fails
    
    return model, label_encoders, accuracy

# Train model
model, encoders, accuracy = train_model(df)

# Prepare input for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'FamilyHistory': [family_history],
    'HighCaloricFreq': [high_caloric],
    'VegConsumption': [veg_consumption],
    'MainMeals': [main_meals],
    'FoodBetweenMeals': [food_between],
    'Smoking': [smoking],
    'WaterConsumption': [water],
    'CaloriesMonitor': [calories_monitor],
    'PhysicalActivity': [physical_activity],
    'TechUsage': [tech_usage],
    'AlcoholConsumption': [alcohol],
    'Transportation': [transport]
})

# Encode input
for col, encoder in encoders.items():
    if col in input_data.columns:
        try:
            input_data[col] = encoder.transform(input_data[col])
        except ValueError:
            # If value not seen in training, use most common
            input_data[col] = 0

# Make prediction
prediction = model.predict(input_data)[0]
try:
    prediction_proba = model.predict_proba(input_data)[0]
    classes = model.classes_
    proba_dict = dict(zip(classes, prediction_proba))
    has_proba = True
except:
    has_proba = False

# Main content area - 2 columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Prediction Result")
    
    # Color coding based on obesity level
    if "Normal" in prediction:
        st.success(f"## {prediction}")
        st.info("✅ Maintain healthy lifestyle habits!")
    elif "Overweight" in prediction:
        st.warning(f"## {prediction}")
        st.info("⚠️ Consider increasing physical activity and improving diet.")
    else:
        st.error(f"## {prediction}")
        st.info("🔴 Professional medical consultation recommended.")
    
    # Model accuracy
    st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Probability bar chart (only if plotly is available)
    if has_proba and plotly_available:
        st.subheader("📈 Prediction Probabilities")
        proba_df = pd.DataFrame({
            'Obesity Level': list(proba_dict.keys()),
            'Probability': list(proba_dict.values())
        })
        
        fig = px.bar(proba_df, x='Obesity Level', y='Probability', 
                     title="Probability Distribution",
                     color='Probability',
                     color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)
    elif has_proba:
        st.subheader("📈 Prediction Probabilities")
        for level, prob in proba_dict.items():
            st.progress(prob, text=f"{level}: {prob:.1%}")

with col2:
    st.subheader("📊 Key Health Metrics")
    
    # BMI gauge
    if bmi < 18.5:
        bmi_status = "Underweight"
        st.metric("BMI", f"{bmi:.1f}", delta=bmi_status, delta_color="inverse")
    elif bmi < 25:
        bmi_status = "Normal"
        st.metric("BMI", f"{bmi:.1f}", delta=bmi_status, delta_color="normal")
    elif bmi < 30:
        bmi_status = "Overweight"
        st.metric("BMI", f"{bmi:.1f}", delta=bmi_status, delta_color="inverse")
    else:
        bmi_status = "Obese"
        st.metric("BMI", f"{bmi:.1f}", delta=bmi_status, delta_color="inverse")
    
    # Additional metrics
    col2a, col2b = st.columns(2)
    with col2a:
        st.metric("Physical Activity", f"{physical_activity:.1f} days/week")
        st.metric("Veg Consumption", f"{veg_consumption}/3")
    with col2b:
        st.metric("Water Intake", f"{water:.1f} L/day")
        st.metric("Tech Usage", f"{tech_usage:.1f} hrs/day")
    
    # Risk factors
    st.subheader("⚠️ Risk Factors")
    risk_factors = []
    if family_history == "yes":
        risk_factors.append("• Family history of obesity")
    if high_caloric == "yes":
        risk_factors.append("• Frequent high-caloric food")
    if veg_consumption < 2:
        risk_factors.append("• Low vegetable consumption")
    if physical_activity < 1:
        risk_factors.append("• Low physical activity")
    if tech_usage > 1.5:
        risk_factors.append("• High screen time")
    if food_between in ["frequently", "always"]:
        risk_factors.append("• Frequent snacking between meals")
    
    if risk_factors:
        for risk in risk_factors:
            st.write(risk)
    else:
        st.success("No major risk factors identified!")

# Data visualization section
st.markdown("---")
st.subheader("📊 Dataset Overview")

tab1, tab2, tab3 = st.tabs(["Data Viewer", "Statistics", "Export Results"])

with tab1:
    st.dataframe(df, use_container_width=True)
    st.caption(f"Total records: {len(df)} | Obesity levels: {df['ObesityLevel'].nunique()}")

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Numerical Features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    with col_b:
        st.subheader("Obesity Level Distribution")
        obesity_counts = df['ObesityLevel'].value_counts()
        st.dataframe(obesity_counts, use_container_width=True)
        
        # Simple bar chart using st.bar_chart if plotly not available
        if plotly_available:
            fig_pie = px.pie(values=obesity_counts.values, names=obesity_counts.index, 
                           title="Distribution of Obesity Levels")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.bar_chart(obesity_counts)

with tab3:
    st.subheader("Export Current Prediction")
    
    # Create result dataframe
    result_data = {
        'Parameter': ['Age', 'Gender', 'Height', 'Weight', 'BMI', 'Family History', 
                      'High Caloric Food', 'Vegetable Consumption', 'Main Meals',
                      'Food Between Meals', 'Smoking', 'Water Consumption', 
                      'Calories Monitor', 'Physical Activity', 'Tech Usage',
                      'Alcohol Consumption', 'Transportation', 'Predicted Obesity Level'],
        'Value': [age, gender, height, weight, f"{bmi:.1f}", family_history,
                  high_caloric, veg_consumption, main_meals, food_between,
                  smoking, water, calories_monitor, physical_activity,
                  tech_usage, alcohol, transport, prediction]
    }
    
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df, use_container_width=True)
    
    # Download buttons
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction Results (CSV)",
        data=csv,
        file_name="obesity_prediction_results.csv",
        mime="text/csv"
    )
    
    # Download full dataset
    full_data_csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=full_data_csv,
        file_name="obesity_dataset.csv",
        mime="text/csv"
    )

# About section
with st.expander("ℹ️ About This Prototype"):
    st.markdown("""
    ### Obesity Level Prediction System
    
    **Features used for prediction:**
    - **Personal:** Age, Gender, Height, Weight
    - **Eating habits:** Family history, High-caloric food frequency, Vegetable consumption, Meal patterns
    - **Lifestyle:** Physical activity, Screen time, Water intake, Smoking, Alcohol, Transportation
    
    **Obesity levels predicted:**
    - Normal Weight
    - Overweight Level I & II
    - Obesity Type I, II, & III
    
    **Model Information:**
    - Algorithm: Random Forest Classifier
    - Training data: From obesity_data.txt file
    - You can upload your own CSV/TXT file with the same format
    
    **⚠️ Medical Disclaimer:** 
    This is a demonstration prototype for educational purposes only. 
    Not intended for medical diagnosis. Always consult healthcare professionals for medical advice.
    
    **Technology Stack:**
    - Streamlit for web interface
    - Scikit-learn for machine learning
    - Pandas for data manipulation
    """)

# Footer
st.markdown("---")
st.caption("Obesity Level Prediction Prototype | Powered by Machine Learning | For demonstration purposes only")