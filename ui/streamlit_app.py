import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import Dict, Any, Tuple, Optional
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
warnings.filterwarnings('ignore')

# Add the src directory to the path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import configuration and prediction utilities
from config import (
    APP_TITLE, APP_DESCRIPTION, VERSION, MODEL_PATH, BACKUP_MODEL_PATH,
    FEATURE_INFO, PAGES, CUSTOM_CSS, RISK_CATEGORIES
)
from prediction_utils import PredictionEngine, PredictionHistory, export_prediction_report

class HeartDiseasePredictor:
    """Main class for the Heart Disease Prediction Streamlit App"""
    
    def __init__(self):
        self.model = None
        self.prediction_engine = None
        self.prediction_history = PredictionHistory()
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
        
        if 'prediction_probabilities' not in st.session_state:
            st.session_state.prediction_probabilities = None
        
        if 'confidence_info' not in st.session_state:
            st.session_state.confidence_info = {}
        
        if 'prediction_interpretation' not in st.session_state:
            st.session_state.prediction_interpretation = {}
        
        if 'risk_explanation' not in st.session_state:
            st.session_state.risk_explanation = ""
        
        if 'population_comparison' not in st.session_state:
            st.session_state.population_comparison = {}
        
        if 'feature_impact' not in st.session_state:
            st.session_state.feature_impact = {}
        
        if 'user_input' not in st.session_state:
            st.session_state.user_input = {}
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Prediction'
        
        if 'show_history' not in st.session_state:
            st.session_state.show_history = False
    
    def load_model_pipeline(self):
        """Load the trained model pipeline with caching for efficiency"""
        try:
            # Try to load the complete pipeline first
            model_path = None
            if os.path.exists(MODEL_PATH):
                model_path = MODEL_PATH
                self.model = joblib.load(MODEL_PATH)
                st.success("✅ Model loaded successfully!")
            elif os.path.exists(BACKUP_MODEL_PATH):
                model_path = BACKUP_MODEL_PATH
                self.model = joblib.load(BACKUP_MODEL_PATH)
                st.warning("⚠️ Loaded backup model. Some features may be limited.")
            else:
                st.error("❌ No trained model found. Please train a model first.")
                return None
            
            # Initialize prediction engine
            self.prediction_engine = PredictionEngine(model_path)
            return self.model
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
        if self.model is not None and self.prediction_engine is not None:
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
                help=FEATURE_INFO['age']['help'],
                key="age_input"
            )
            
            # Sex
            sex_option = st.selectbox(
                FEATURE_INFO['sex']['label'],
                list(FEATURE_INFO['sex']['options'].keys()),
                index=list(FEATURE_INFO['sex']['options'].keys()).index(FEATURE_INFO['sex']['default']),
                help=FEATURE_INFO['sex']['help'],
                key="sex_input"
            )
            input_data['sex'] = FEATURE_INFO['sex']['options'][sex_option]
            
            # Chest Pain Type
            cp_option = st.selectbox(
                FEATURE_INFO['cp']['label'],
                list(FEATURE_INFO['cp']['options'].keys()),
                index=list(FEATURE_INFO['cp']['options'].keys()).index(FEATURE_INFO['cp']['default']),
                help=FEATURE_INFO['cp']['help'],
                key="cp_input"
            )
            input_data['cp'] = FEATURE_INFO['cp']['options'][cp_option]
            
            # Resting Blood Pressure
            input_data['trestbps'] = st.number_input(
                FEATURE_INFO['trestbps']['label'],
                min_value=FEATURE_INFO['trestbps']['min_val'],
                max_value=FEATURE_INFO['trestbps']['max_val'],
                value=FEATURE_INFO['trestbps']['default'],
                help=FEATURE_INFO['trestbps']['help'],
                key="trestbps_input"
            )
            
            # Cholesterol
            input_data['chol'] = st.number_input(
                FEATURE_INFO['chol']['label'],
                min_value=FEATURE_INFO['chol']['min_val'],
                max_value=FEATURE_INFO['chol']['max_val'],
                value=FEATURE_INFO['chol']['default'],
                help=FEATURE_INFO['chol']['help'],
                key="chol_input"
            )
            
            # Fasting Blood Sugar
            fbs_option = st.selectbox(
                FEATURE_INFO['fbs']['label'],
                list(FEATURE_INFO['fbs']['options'].keys()),
                index=list(FEATURE_INFO['fbs']['options'].keys()).index(FEATURE_INFO['fbs']['default']),
                help=FEATURE_INFO['fbs']['help'],
                key="fbs_input"
            )
            input_data['fbs'] = FEATURE_INFO['fbs']['options'][fbs_option]
            
            # Resting ECG
            restecg_option = st.selectbox(
                FEATURE_INFO['restecg']['label'],
                list(FEATURE_INFO['restecg']['options'].keys()),
                index=list(FEATURE_INFO['restecg']['options'].keys()).index(FEATURE_INFO['restecg']['default']),
                help=FEATURE_INFO['restecg']['help'],
                key="restecg_input"
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
                help=FEATURE_INFO['thalach']['help'],
                key="thalach_input"
            )
            
            # Exercise Induced Angina
            exang_option = st.selectbox(
                FEATURE_INFO['exang']['label'],
                list(FEATURE_INFO['exang']['options'].keys()),
                index=list(FEATURE_INFO['exang']['options'].keys()).index(FEATURE_INFO['exang']['default']),
                help=FEATURE_INFO['exang']['help'],
                key="exang_input"
            )
            input_data['exang'] = FEATURE_INFO['exang']['options'][exang_option]
            
            # ST Depression
            input_data['oldpeak'] = st.number_input(
                FEATURE_INFO['oldpeak']['label'],
                min_value=FEATURE_INFO['oldpeak']['min_val'],
                max_value=FEATURE_INFO['oldpeak']['max_val'],
                value=FEATURE_INFO['oldpeak']['default'],
                step=FEATURE_INFO['oldpeak']['step'],
                help=FEATURE_INFO['oldpeak']['help'],
                key="oldpeak_input"
            )
            
            # ST Slope
            slope_option = st.selectbox(
                FEATURE_INFO['slope']['label'],
                list(FEATURE_INFO['slope']['options'].keys()),
                index=list(FEATURE_INFO['slope']['options'].keys()).index(FEATURE_INFO['slope']['default']),
                help=FEATURE_INFO['slope']['help'],
                key="slope_input"
            )
            input_data['slope'] = FEATURE_INFO['slope']['options'][slope_option]
            
            # Major Vessels
            ca_option = st.selectbox(
                FEATURE_INFO['ca']['label'],
                list(FEATURE_INFO['ca']['options'].keys()),
                index=list(FEATURE_INFO['ca']['options'].keys()).index(FEATURE_INFO['ca']['default']),
                help=FEATURE_INFO['ca']['help'],
                key="ca_input"
            )
            input_data['ca'] = FEATURE_INFO['ca']['options'][ca_option]
            
            # Thalassemia
            thal_option = st.selectbox(
                FEATURE_INFO['thal']['label'],
                list(FEATURE_INFO['thal']['options'].keys()),
                index=list(FEATURE_INFO['thal']['options'].keys()).index(FEATURE_INFO['thal']['default']),
                help=FEATURE_INFO['thal']['help'],
                key="thal_input"
            )
            input_data['thal'] = FEATURE_INFO['thal']['options'][thal_option]
        
        return input_data
    
    def make_prediction(self, input_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray]]:
        """Make prediction using the enhanced prediction engine"""
        if self.prediction_engine is None:
            st.error("❌ Prediction engine not available")
            return None, None, None
        
        return self.prediction_engine.make_prediction(input_data)
    
    def display_prediction_results(self, prediction: int, confidence: float, probabilities: Optional[np.ndarray], input_data: Dict[str, Any]):
        """Display comprehensive prediction results with clear visual indicators and explanations"""
        if prediction is None or self.prediction_engine is None:
            return
        
        # Calculate detailed confidence information
        confidence_info = self.prediction_engine.calculate_prediction_confidence(probabilities)
        st.session_state.confidence_info = confidence_info
        
        # Interpret prediction result
        prediction_interpretation = self.prediction_engine.interpret_prediction_result(prediction, confidence, input_data)
        st.session_state.prediction_interpretation = prediction_interpretation
        
        # Generate risk explanation
        risk_explanation = self.prediction_engine.generate_risk_explanation(input_data, prediction_interpretation)
        st.session_state.risk_explanation = risk_explanation
        
        # Get population comparison
        population_comparison = self.prediction_engine.compare_with_population(input_data)
        st.session_state.population_comparison = population_comparison
        
        # Get feature impact analysis
        feature_impact = self.prediction_engine.get_feature_impact_analysis(input_data)
        st.session_state.feature_impact = feature_impact
        
        # Display main prediction result
        result_class = "low-risk" if prediction == 0 else "high-risk"
        icon = prediction_interpretation.get('icon', '🎯')
        
        st.markdown(
            f'<div class="prediction-result {result_class}">'
            f'{icon} Prediction: {prediction_interpretation["risk_category"]}<br>'
            f'📊 Confidence: {confidence:.1%} ({confidence_info["confidence_level"]})'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Create tabs for detailed information
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Assessment", "📊 Confidence", "👥 Population Comparison", 
            "🔍 Feature Impact", "📄 Full Report"
        ])
        
        with tab1:
            st.markdown("### Risk Assessment")
            st.markdown(risk_explanation)
        
        with tab2:
            self.display_prediction_confidence(confidence_info, probabilities)
        
        with tab3:
            self.display_population_comparison(population_comparison)
        
        with tab4:
            self.display_feature_impact_analysis(feature_impact, input_data)
        
        with tab5:
            self.display_full_report(input_data, prediction_interpretation, confidence_info, risk_explanation)
        
        # Save prediction to history
        self.prediction_history.save_prediction(input_data, prediction, confidence)
    
    def display_prediction_confidence(self, confidence_info: Dict[str, Any], probabilities: Optional[np.ndarray]):
        """Create prediction confidence visualization with progress bars and gauges"""
        st.markdown("### Prediction Confidence Analysis")
        
        # Confidence level indicator
        confidence_score = confidence_info.get('confidence_score', 0.5)
        
        # Create confidence gauge using plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence interpretation
        st.markdown(f"**Confidence Level:** {confidence_info.get('confidence_level', 'Unknown')}")
        st.markdown(f"**Interpretation:** {confidence_info.get('certainty_description', 'No description available')}")
        
        # Probability breakdown if available
        if probabilities is not None and len(probabilities) > 1:
            st.markdown("### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Risk Level': ['Low Risk', 'High Risk'],
                'Probability': probabilities
            })
            
            fig_bar = px.bar(prob_df, x='Risk Level', y='Probability', 
                           title='Prediction Probabilities',
                           color='Probability',
                           color_continuous_scale='RdYlGn_r')
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def display_population_comparison(self, population_comparison: Dict[str, Any]):
        """Show user's risk relative to dataset statistics"""
        st.markdown("### Population Comparison")
        
        if not population_comparison:
            st.info("Population comparison data not available")
            return
        
        # Create comparison visualizations
        for param, data in population_comparison.items():
            if isinstance(data, dict) and 'percentile' in data:
                st.markdown(f"**{param.replace('_', ' ').title()}:**")
                
                # Create percentile visualization
                percentile = data['percentile']
                
                # Progress bar for percentile
                st.progress(percentile / 100)
                st.markdown(f"Your value: {data['value']} (at {percentile:.0f}th percentile)")
                st.markdown(f"*{data['interpretation']}*")
                st.markdown("---")
    
    def display_feature_impact_analysis(self, feature_impact: Dict[str, Any], input_data: Dict[str, Any]):
        """Show which features most influence the prediction"""
        st.markdown("### Feature Impact Analysis")
        
        if not feature_impact:
            st.info("Feature impact analysis not available for this model type")
            return
        
        # Sort features by importance
        sorted_features = sorted(feature_impact.items(), key=lambda x: x[1].get('importance', 0), reverse=True)
        
        # Display top influential features
        st.markdown("**Most Influential Features:**")
        
        impact_data = []
        for feature, impact_info in sorted_features[:8]:  # Show top 8 features
            importance = impact_info.get('importance', 0)
            impact_level = impact_info.get('impact_level', 'Unknown')
            user_value = impact_info.get('user_value', 'N/A')
            
            # Get feature label from config
            feature_label = FEATURE_INFO.get(feature, {}).get('label', feature)
            
            impact_data.append({
                'Feature': feature_label,
                'Your Value': user_value,
                'Impact Level': impact_level,
                'Importance Score': importance
            })
        
        if impact_data:
            impact_df = pd.DataFrame(impact_data)
            
            # Create horizontal bar chart
            fig = px.bar(impact_df, x='Importance Score', y='Feature', 
                        orientation='h', color='Impact Level',
                        title='Feature Importance in Your Prediction',
                        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(impact_df, use_container_width=True)
    
    def display_full_report(self, input_data: Dict[str, Any], prediction_interpretation: Dict[str, Any], 
                          confidence_info: Dict[str, Any], risk_explanation: str):
        """Display full report and export functionality"""
        st.markdown("### Complete Assessment Report")
        
        # Generate report
        report_text = export_prediction_report(input_data, prediction_interpretation, confidence_info, risk_explanation)
        
        # Display report
        st.markdown(report_text)
        
        # Download button
        st.download_button(
            label="📄 Download Report",
            data=report_text,
            file_name=f"heart_disease_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            help="Download a comprehensive report of your heart disease risk assessment",
            key="download_report_button"
        )
    
    def display_prediction_history(self):
        """Display previous predictions"""
        st.markdown("### Prediction History")
        
        recent_predictions = self.prediction_history.get_recent_predictions(10)
        
        if not recent_predictions:
            st.info("No previous predictions found")
            return
        
        # Create history table
        history_data = []
        for pred in recent_predictions:
            timestamp = datetime.fromisoformat(pred['timestamp']).strftime('%Y-%m-%d %H:%M')
            risk_level = 'High Risk' if pred['prediction'] == 1 else 'Low Risk'
            confidence = f"{pred['confidence']:.1%}"
            
            history_data.append({
                'Date': timestamp,
                'Risk Level': risk_level,
                'Confidence': confidence,
                'Age': pred['input_data'].get('age', 'N/A'),
                'Sex': 'Male' if pred['input_data'].get('sex', 0) == 1 else 'Female'
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History", 
                    help="Delete all stored prediction history",
                    key="clear_history_button"):
            self.prediction_history.clear_history()
            st.success("Prediction history cleared!")
            try:
                st.experimental_rerun()
            except AttributeError:
                # For newer versions of Streamlit
                st.rerun()
    
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
            if st.button("🔍 Predict Risk", type="primary", use_container_width=True, 
                        help="Click to analyze your medical data and get heart disease risk prediction"):
                # Validate input
                is_valid, message = self.validate_user_input(input_data)
                
                if is_valid:
                    with st.spinner("Analyzing your data..."):
                        prediction, confidence, probabilities = self.make_prediction(input_data)
                        
                        if prediction is not None:
                            st.session_state.prediction_made = True
                            st.session_state.prediction_result = prediction
                            st.session_state.prediction_confidence = confidence
                            st.session_state.prediction_probabilities = probabilities
                            st.session_state.user_input = input_data
                else:
                    st.error(f"❌ Input validation failed: {message}")
        
        # Real-time prediction updates (optional feature)
        enable_realtime = st.checkbox("🔄 Enable Real-time Updates", 
                                     help="Update predictions as you modify input values",
                                     key="realtime_checkbox")
        if enable_realtime:
            if self.prediction_engine is not None:
                is_valid, _ = self.validate_user_input(input_data)
                if is_valid:
                    prediction, confidence, probabilities = self.make_prediction(input_data)
                    if prediction is not None:
                        st.session_state.prediction_made = True
                        st.session_state.prediction_result = prediction
                        st.session_state.prediction_confidence = confidence
                        st.session_state.prediction_probabilities = probabilities
                        st.session_state.user_input = input_data
        
        # Display results if prediction was made
        if st.session_state.prediction_made:
            st.markdown("---")
            st.subheader("📋 Comprehensive Prediction Results")
            self.display_prediction_results(
                st.session_state.prediction_result,
                st.session_state.prediction_confidence,
                st.session_state.prediction_probabilities,
                st.session_state.user_input
            )
        
        # Prediction history section
        st.markdown("---")
        if st.button("📊 View Prediction History", 
                    help="Show your previous predictions and analysis history",
                    key="history_button"):
            st.session_state.show_history = not st.session_state.show_history
        
        if st.session_state.show_history:
            self.display_prediction_history()
    
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
        # Set page configuration with accessibility improvements
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon="❤️",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/issues',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': f"{APP_TITLE} - AI-powered heart disease risk assessment"
            }
        )
        
        # Add security headers via HTML meta tags
        st.markdown("""
        <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';">
        <meta http-equiv="X-Content-Type-Options" content="nosniff">
        <meta http-equiv="X-Frame-Options" content="DENY">
        <meta http-equiv="X-XSS-Protection" content="1; mode=block">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta charset="utf-8">
        """, unsafe_allow_html=True)
        
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