import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import configuration
from config import (
    APP_TITLE, APP_DESCRIPTION, VERSION, MODEL_PATH, BACKUP_MODEL_PATH,
    FEATURE_INFO, PAGES, CUSTOM_CSS, RISK_CATEGORIES
)

class HeartDiseasePredictor:
    """Main class for the Heart Disease Prediction Streamlit App"""
    
    def __init__(self):
        self.model = None
        self.initialize_session_state()
        self.load_model_pipeline()
    
    def initialize_session_state(self):
        """Initialize session state variables for maintaining user interaction state"""
        if 'prediction_made' not in st.session_state:
            st.session_state.prediction_made = False
        
        if 'prediction_result' not in st.session_state:
            st.session_state.prediction_result = None
        
        if 'prediction_confidence' not in st.session_state:
            st.session_state.prediction_confidence = None
        
        if 'user_input' not in st.session_state:
            st.session_state.user_input = {}
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Prediction'
    
    @st.cache_resource
    def load_model_pipeline(_self):
        """Load the trained model pipeline with caching for efficiency"""
        try:
            # Try to load the complete pipeline first
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                st.success("✅ Model loaded successfully!")
                return model
            elif os.path.exists(BACKUP_MODEL_PATH):
                model = joblib.load(BACKUP_MODEL_PATH)
                st.warning("⚠️ Loaded backup model. Some features may be limited.")
                return model
            else:
                st.error("❌ No trained model found. Please train a model first.")
                return None
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None
    
    def create_app_header(self):
        """Create the main application header with project information"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"**{APP_DESCRIPTION}**")
            st.markdown(f"*Version {VERSION}*")
        
        st.markdown("---")
        
        # Add instructions
        with st.expander("ℹ️ How to use this application"):
            st.markdown("""
            **Welcome to the Heart Disease Risk Prediction Tool!**
            
            This application uses machine learning to assess heart disease risk based on medical parameters.
            
            **Instructions:**
            1. **Prediction Page**: Enter your medical information to get a risk assessment
            2. **Data Explorer**: Explore the dataset and understand the patterns
            3. **Model Info**: Learn about the machine learning models used
            
            **Important Note:** This tool is for educational purposes only and should not replace professional medical advice.
            Always consult with healthcare professionals for medical decisions.
            """)
    
    def create_sidebar_navigation(self) -> str:
        """Create sidebar navigation with multiple pages"""
        st.sidebar.title("🧭 Navigation")
        
        # Page selection
        selected_page = st.sidebar.selectbox(
            "Select a page:",
            list(PAGES.keys()),
            format_func=lambda x: PAGES[x],
            key="page_selector"
        )
        
        st.sidebar.markdown("---")
        
        # Add some information in sidebar
        st.sidebar.markdown("### 📋 Quick Info")
        st.sidebar.info(
            "This app analyzes 13 medical parameters to predict heart disease risk using "
            "machine learning algorithms trained on the UCI Heart Disease dataset."
        )
        
        # Model status
        if self.model is not None:
            st.sidebar.success("🤖 Model: Ready")
        else:
            st.sidebar.error("🤖 Model: Not Available")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚠️ Disclaimer")
        st.sidebar.warning(
            "This tool is for educational purposes only. "
            "Always consult healthcare professionals for medical advice."
        )
        
        return selected_page
    
    def create_feature_descriptions(self):
        """Create feature descriptions with tooltips"""
        with st.expander("📖 Medical Parameter Descriptions"):
            st.markdown("**Understanding the Medical Parameters:**")
            
            for feature, info in FEATURE_INFO.items():
                st.markdown(f"**{info['label']}**: {info['description']}")
                if 'help' in info:
                    st.markdown(f"*{info['help']}*")
                st.markdown("")
    
    def validate_user_input(self, input_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate user input with range checking and data type validation"""
        try:
            for feature, value in input_data.items():
                if feature not in FEATURE_INFO:
                    return False, f"Unknown feature: {feature}"
                
                feature_info = FEATURE_INFO[feature]
                
                # Check for numeric features
                if 'min_val' in feature_info and 'max_val' in feature_info:
                    if not isinstance(value, (int, float)):
                        return False, f"{feature_info['label']} must be a number"
                    
                    if value < feature_info['min_val'] or value > feature_info['max_val']:
                        return False, (
                            f"{feature_info['label']} must be between "
                            f"{feature_info['min_val']} and {feature_info['max_val']}"
                        )
                
                # Check for categorical features
                elif 'options' in feature_info:
                    if value not in feature_info['options'].values():
                        return False, f"Invalid value for {feature_info['label']}"
            
            return True, "All inputs are valid"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def render_input_form(self) -> Dict[str, Any]:
        """Render input form with all 14 heart disease feature inputs and validation"""
        st.subheader("🏥 Enter Your Medical Information")
        
        input_data = {}
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        # Column 1 features
        with col1:
            st.markdown("**Basic Information**")
            
            # Age
            input_data['age'] = st.number_input(
                FEATURE_INFO['age']['label'],
                min_value=FEATURE_INFO['age']['min_val'],
                max_value=FEATURE_INFO['age']['max_val'],
                value=FEATURE_INFO['age']['default'],
                help=FEATURE_INFO['age']['help']
            )
            
            # Sex
            sex_option = st.selectbox(
                FEATURE_INFO['sex']['label'],
                list(FEATURE_INFO['sex']['options'].keys()),
                index=list(FEATURE_INFO['sex']['options'].keys()).index(FEATURE_INFO['sex']['default']),
                help=FEATURE_INFO['sex']['help']
            )
            input_data['sex'] = FEATURE_INFO['sex']['options'][sex_option]
            
            # Chest Pain Type
            cp_option = st.selectbox(
                FEATURE_INFO['cp']['label'],
                list(FEATURE_INFO['cp']['options'].keys()),
                index=list(FEATURE_INFO['cp']['options'].keys()).index(FEATURE_INFO['cp']['default']),
                help=FEATURE_INFO['cp']['help']
            )
            input_data['cp'] = FEATURE_INFO['cp']['options'][cp_option]
            
            # Resting Blood Pressure
            input_data['trestbps'] = st.number_input(
                FEATURE_INFO['trestbps']['label'],
                min_value=FEATURE_INFO['trestbps']['min_val'],
                max_value=FEATURE_INFO['trestbps']['max_val'],
                value=FEATURE_INFO['trestbps']['default'],
                help=FEATURE_INFO['trestbps']['help']
            )
            
            # Cholesterol
            input_data['chol'] = st.number_input(
                FEATURE_INFO['chol']['label'],
                min_value=FEATURE_INFO['chol']['min_val'],
                max_value=FEATURE_INFO['chol']['max_val'],
                value=FEATURE_INFO['chol']['default'],
                help=FEATURE_INFO['chol']['help']
            )
            
            # Fasting Blood Sugar
            fbs_option = st.selectbox(
                FEATURE_INFO['fbs']['label'],
                list(FEATURE_INFO['fbs']['options'].keys()),
                index=list(FEATURE_INFO['fbs']['options'].keys()).index(FEATURE_INFO['fbs']['default']),
                help=FEATURE_INFO['fbs']['help']
            )
            input_data['fbs'] = FEATURE_INFO['fbs']['options'][fbs_option]
            
            # Resting ECG
            restecg_option = st.selectbox(
                FEATURE_INFO['restecg']['label'],
                list(FEATURE_INFO['restecg']['options'].keys()),
                index=list(FEATURE_INFO['restecg']['options'].keys()).index(FEATURE_INFO['restecg']['default']),
                help=FEATURE_INFO['restecg']['help']
            )
            input_data['restecg'] = FEATURE_INFO['restecg']['options'][restecg_option]
        
        # Column 2 features
        with col2:
            st.markdown("**Exercise & Heart Parameters**")
            
            # Maximum Heart Rate
            input_data['thalach'] = st.number_input(
                FEATURE_INFO['thalach']['label'],
                min_value=FEATURE_INFO['thalach']['min_val'],
                max_value=FEATURE_INFO['thalach']['max_val'],
                value=FEATURE_INFO['thalach']['default'],
                help=FEATURE_INFO['thalach']['help']
            )
            
            # Exercise Induced Angina
            exang_option = st.selectbox(
                FEATURE_INFO['exang']['label'],
                list(FEATURE_INFO['exang']['options'].keys()),
                index=list(FEATURE_INFO['exang']['options'].keys()).index(FEATURE_INFO['exang']['default']),
                help=FEATURE_INFO['exang']['help']
            )
            input_data['exang'] = FEATURE_INFO['exang']['options'][exang_option]
            
            # ST Depression
            input_data['oldpeak'] = st.number_input(
                FEATURE_INFO['oldpeak']['label'],
                min_value=FEATURE_INFO['oldpeak']['min_val'],
                max_value=FEATURE_INFO['oldpeak']['max_val'],
                value=FEATURE_INFO['oldpeak']['default'],
                step=FEATURE_INFO['oldpeak']['step'],
                help=FEATURE_INFO['oldpeak']['help']
            )
            
            # ST Slope
            slope_option = st.selectbox(
                FEATURE_INFO['slope']['label'],
                list(FEATURE_INFO['slope']['options'].keys()),
                index=list(FEATURE_INFO['slope']['options'].keys()).index(FEATURE_INFO['slope']['default']),
                help=FEATURE_INFO['slope']['help']
            )
            input_data['slope'] = FEATURE_INFO['slope']['options'][slope_option]
            
            # Major Vessels
            ca_option = st.selectbox(
                FEATURE_INFO['ca']['label'],
                list(FEATURE_INFO['ca']['options'].keys()),
                index=list(FEATURE_INFO['ca']['options'].keys()).index(FEATURE_INFO['ca']['default']),
                help=FEATURE_INFO['ca']['help']
            )
            input_data['ca'] = FEATURE_INFO['ca']['options'][ca_option]
            
            # Thalassemia
            thal_option = st.selectbox(
                FEATURE_INFO['thal']['label'],
                list(FEATURE_INFO['thal']['options'].keys()),
                index=list(FEATURE_INFO['thal']['options'].keys()).index(FEATURE_INFO['thal']['default']),
                help=FEATURE_INFO['thal']['help']
            )
            input_data['thal'] = FEATURE_INFO['thal']['options'][thal_option]
        
        return input_data
    
    def make_prediction(self, input_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[float]]:
        """Make prediction using the loaded model"""
        if self.model is None:
            st.error("❌ Model not available for prediction")
            return None, None
        
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.model.predict(input_df)[0]
            
            # Get prediction probability if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(input_df)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.8  # Default confidence if probabilities not available
            
            return int(prediction), float(confidence)
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None
    
    def display_prediction_result(self, prediction: int, confidence: float):
        """Display prediction result with styling"""
        if prediction is None:
            return
        
        risk_info = RISK_CATEGORIES[prediction]
        
        # Create result container
        result_class = "low-risk" if prediction == 0 else "high-risk"
        
        st.markdown(
            f'<div class="prediction-result {result_class}">'
            f'🎯 Prediction: {risk_info["label"]}<br>'
            f'📊 Confidence: {confidence:.1%}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Add description and recommendations
        st.markdown(f"**Assessment:** {risk_info['description']}")
        
        st.markdown("**Recommendations:**")
        for rec in risk_info['recommendations']:
            st.markdown(f"• {rec}")
        
        # Add confidence interpretation
        if confidence >= 0.8:
            st.info("🎯 High confidence prediction")
        elif confidence >= 0.6:
            st.warning("⚠️ Moderate confidence prediction")
        else:
            st.warning("⚠️ Low confidence prediction - consider consulting a healthcare professional")
    
    def render_prediction_page(self):
        """Render the main prediction interface"""
        st.title("🔮 Heart Disease Risk Prediction")
        
        # Create feature descriptions
        self.create_feature_descriptions()
        
        # Render input form
        input_data = self.render_input_form()
        
        # Validate and make prediction
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
                # Validate input
                is_valid, message = self.validate_user_input(input_data)
                
                if is_valid:
                    with st.spinner("Analyzing your data..."):
                        prediction, confidence = self.make_prediction(input_data)
                        
                        if prediction is not None:
                            st.session_state.prediction_made = True
                            st.session_state.prediction_result = prediction
                            st.session_state.prediction_confidence = confidence
                            st.session_state.user_input = input_data
                else:
                    st.error(f"❌ Input validation failed: {message}")
        
        # Display results if prediction was made
        if st.session_state.prediction_made:
            st.markdown("---")
            st.subheader("📋 Prediction Results")
            self.display_prediction_result(
                st.session_state.prediction_result,
                st.session_state.prediction_confidence
            )
    
    def render_data_explorer_page(self):
        """Render data explorer page (placeholder for now)"""
        st.title("📊 Data Explorer")
        st.info("🚧 Data exploration features will be implemented in the next phase.")
        
        # Show some basic information
        st.markdown("""
        **Coming Soon:**
        - Interactive data visualizations
        - Feature correlation analysis
        - Dataset statistics and insights
        - Population risk patterns
        """)
    
    def render_model_info_page(self):
        """Render model information page"""
        st.title("🤖 Model Information")
        
        st.markdown("""
        ### About the Model
        
        This heart disease prediction system uses machine learning algorithms trained on the 
        **UCI Heart Disease Dataset** (Cleveland database).
        
        **Dataset Information:**
        - **Source**: UCI Machine Learning Repository
        - **Instances**: 303 patients
        - **Features**: 13 medical parameters
        - **Target**: Heart disease presence (binary classification)
        
        **Features Used:**
        1. **Age**: Age in years
        2. **Sex**: Gender (1 = male, 0 = female)
        3. **CP**: Chest pain type (4 values)
        4. **Trestbps**: Resting blood pressure
        5. **Chol**: Serum cholesterol in mg/dl
        6. **FBS**: Fasting blood sugar > 120 mg/dl
        7. **Restecg**: Resting electrocardiographic results
        8. **Thalach**: Maximum heart rate achieved
        9. **Exang**: Exercise induced angina
        10. **Oldpeak**: ST depression induced by exercise
        11. **Slope**: Slope of the peak exercise ST segment
        12. **CA**: Number of major vessels colored by fluoroscopy
        13. **Thal**: Thalassemia type
        
        ### Model Performance
        The model has been trained and optimized using various machine learning algorithms
        including Logistic Regression, Random Forest, and Support Vector Machines.
        
        ### Important Notes
        - This model is for educational and research purposes only
        - Results should not be used as a substitute for professional medical diagnosis
        - Always consult with healthcare professionals for medical decisions
        """)
        
        if self.model is not None:
            st.success("✅ Model is loaded and ready for predictions")
        else:
            st.error("❌ Model is not available")
    
    def run(self):
        """Main application runner"""
        # Set page configuration
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon="❤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Create header
        self.create_app_header()
        
        # Create navigation
        selected_page = self.create_sidebar_navigation()
        
        # Render selected page
        if selected_page == 'Prediction':
            self.render_prediction_page()
        elif selected_page == 'Data Explorer':
            self.render_data_explorer_page()
        elif selected_page == 'Model Info':
            self.render_model_info_page()

def main():
    """Main function to run the Streamlit app"""
    app = HeartDiseasePredictor()
    app.run()

if __name__ == "__main__":
    main()