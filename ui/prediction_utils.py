"""
Prediction utilities for the Heart Disease Prediction Streamlit App
Contains helper functions for prediction processing, confidence calculation, and result interpretation
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple, List, Optional
import json
import os
from datetime import datetime
import joblib

class PredictionEngine:
    """Enhanced prediction engine with confidence scoring and result interpretation"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.load_model()
        self.load_population_stats()
    
    def load_model(self):
        """Load the trained model pipeline"""
        try:
            # Normalize the path
            model_path = os.path.normpath(self.model_path)
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print(f"✅ Model loaded successfully from: {model_path}")
            else:
                # Try alternative paths
                alt_paths = [
                    os.path.join("models", "production", "complete_pipeline.pkl"),
                    os.path.join("models", "production", "final_model.pkl"),
                    os.path.join("..", "models", "production", "complete_pipeline.pkl"),
                    os.path.join("..", "models", "production", "final_model.pkl")
                ]
                
                model_loaded = False
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        self.model = joblib.load(alt_path)
                        self.model_path = alt_path
                        print(f"✅ Model loaded from alternative path: {alt_path}")
                        model_loaded = True
                        break
                
                if not model_loaded:
                    print(f"❌ Model file not found at: {model_path}")
                    print(f"❌ Also tried: {alt_paths}")
                    
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_population_stats(self):
        """Load population statistics for comparison"""
        # Default population statistics (these would normally come from the training data)
        self.population_stats = {
            'age': {'mean': 54.4, 'std': 9.0},
            'trestbps': {'mean': 131.6, 'std': 17.5},
            'chol': {'mean': 246.3, 'std': 51.8},
            'thalach': {'mean': 149.6, 'std': 22.9},
            'oldpeak': {'mean': 1.04, 'std': 1.16},
            'heart_disease_prevalence': 0.54  # 54% in the dataset
        }
    
    def preprocess_user_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Format input data for model pipeline"""
        try:
            # Create DataFrame with correct feature order
            processed_data = {}
            for feature in self.feature_names:
                if feature in input_data:
                    processed_data[feature] = input_data[feature]
                else:
                    # Set default values for missing features
                    processed_data[feature] = 0
            
            # Convert to DataFrame
            df = pd.DataFrame([processed_data])
            
            # Ensure correct data types
            numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            for feature in numeric_features:
                if feature in df.columns:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
            
            return df
        
        except Exception as e:
            st.error(f"Error preprocessing input: {str(e)}")
            return None
    
    def make_prediction(self, input_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray]]:
        """Make prediction with confidence scoring"""
        if self.model is None:
            return None, None, None
        
        try:
            # Preprocess input
            processed_df = self.preprocess_user_input(input_data)
            if processed_df is None:
                return None, None, None
            
            # Make prediction
            prediction = self.model.predict(processed_df)[0]
            
            # Get prediction probabilities
            probabilities = None
            confidence = 0.5  # Default confidence
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_df)[0]
                confidence = max(probabilities)
            elif hasattr(self.model, 'decision_function'):
                # For SVM models
                decision_scores = self.model.decision_function(processed_df)[0]
                # Convert decision score to probability-like confidence
                confidence = 1 / (1 + np.exp(-abs(decision_scores)))
            
            return int(prediction), float(confidence), probabilities
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, None
    
    def calculate_prediction_confidence(self, probabilities: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate detailed prediction confidence metrics"""
        if probabilities is None:
            return {
                'confidence_level': 'Unknown',
                'confidence_score': 0.5,
                'certainty_description': 'Confidence information not available'
            }
        
        max_prob = max(probabilities)
        prob_diff = abs(probabilities[1] - probabilities[0]) if len(probabilities) > 1 else 0
        
        # Determine confidence level
        if max_prob >= 0.9:
            confidence_level = 'Very High'
            certainty_description = 'The model is very confident in this prediction'
        elif max_prob >= 0.8:
            confidence_level = 'High'
            certainty_description = 'The model is confident in this prediction'
        elif max_prob >= 0.7:
            confidence_level = 'Moderate'
            certainty_description = 'The model has moderate confidence in this prediction'
        elif max_prob >= 0.6:
            confidence_level = 'Low'
            certainty_description = 'The model has low confidence - consider additional evaluation'
        else:
            confidence_level = 'Very Low'
            certainty_description = 'The model has very low confidence - seek professional medical advice'
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': max_prob,
            'probability_difference': prob_diff,
            'certainty_description': certainty_description,
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
    
    def interpret_prediction_result(self, prediction: int, confidence: float, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret prediction result with risk level categorization"""
        risk_levels = {
            0: {
                'risk_category': 'Low Risk',
                'risk_level': 'Low',
                'color': 'green',
                'icon': '✅',
                'description': 'Based on the provided medical parameters, you appear to have a low risk of heart disease.',
                'detailed_explanation': 'Your medical parameters fall within ranges typically associated with lower cardiovascular risk.'
            },
            1: {
                'risk_category': 'High Risk',
                'risk_level': 'High',
                'color': 'red',
                'icon': '⚠️',
                'description': 'Based on the provided medical parameters, you may have an elevated risk of heart disease.',
                'detailed_explanation': 'Several of your medical parameters suggest increased cardiovascular risk that warrants medical attention.'
            }
        }
        
        base_result = risk_levels.get(prediction, risk_levels[0])
        
        # Add confidence-based modifications
        if confidence < 0.6:
            base_result['description'] += ' However, the prediction confidence is low, so additional medical evaluation is recommended.'
        
        # Add personalized risk factors
        risk_factors = self.identify_risk_factors(input_data)
        base_result['risk_factors'] = risk_factors
        
        return base_result
    
    def identify_risk_factors(self, input_data: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors from user input"""
        risk_factors = []
        
        # Age risk
        if input_data.get('age', 0) > 65:
            risk_factors.append('Advanced age (>65 years)')
        elif input_data.get('age', 0) > 55:
            risk_factors.append('Elevated age (>55 years)')
        
        # Gender risk
        if input_data.get('sex', 0) == 1:  # Male
            risk_factors.append('Male gender (higher baseline risk)')
        
        # Blood pressure
        if input_data.get('trestbps', 0) > 140:
            risk_factors.append('High blood pressure (>140 mmHg)')
        elif input_data.get('trestbps', 0) > 130:
            risk_factors.append('Elevated blood pressure (>130 mmHg)')
        
        # Cholesterol
        if input_data.get('chol', 0) > 240:
            risk_factors.append('High cholesterol (>240 mg/dl)')
        elif input_data.get('chol', 0) > 200:
            risk_factors.append('Borderline high cholesterol (>200 mg/dl)')
        
        # Chest pain
        if input_data.get('cp', 0) in [1, 2]:  # Typical or atypical angina
            risk_factors.append('Chest pain symptoms')
        
        # Exercise capacity
        max_hr_expected = 220 - input_data.get('age', 50)
        if input_data.get('thalach', 150) < max_hr_expected * 0.7:
            risk_factors.append('Reduced exercise capacity')
        
        # Exercise-induced symptoms
        if input_data.get('exang', 0) == 1:
            risk_factors.append('Exercise-induced chest pain')
        
        # ST depression
        if input_data.get('oldpeak', 0) > 2.0:
            risk_factors.append('Significant ST depression')
        
        # Fasting blood sugar
        if input_data.get('fbs', 0) == 1:
            risk_factors.append('Elevated fasting blood sugar')
        
        return risk_factors
    
    def generate_risk_explanation(self, input_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> str:
        """Generate personalized health insights and explanations"""
        explanation_parts = []
        
        # Start with overall assessment
        explanation_parts.append(f"**Overall Assessment:** {prediction_result['description']}")
        
        # Add risk factors if any
        if prediction_result.get('risk_factors'):
            explanation_parts.append("\n**Identified Risk Factors:**")
            for factor in prediction_result['risk_factors']:
                explanation_parts.append(f"• {factor}")
        
        # Add protective factors
        protective_factors = self.identify_protective_factors(input_data)
        if protective_factors:
            explanation_parts.append("\n**Protective Factors:**")
            for factor in protective_factors:
                explanation_parts.append(f"• {factor}")
        
        # Add recommendations based on risk level
        if prediction_result['risk_level'] == 'High':
            explanation_parts.append("\n**Immediate Recommendations:**")
            explanation_parts.append("• Consult with a cardiologist or healthcare provider")
            explanation_parts.append("• Consider cardiac stress testing or imaging")
            explanation_parts.append("• Review and optimize medications")
            explanation_parts.append("• Implement lifestyle modifications immediately")
        else:
            explanation_parts.append("\n**Preventive Recommendations:**")
            explanation_parts.append("• Maintain regular exercise routine")
            explanation_parts.append("• Follow heart-healthy diet")
            explanation_parts.append("• Regular health screenings")
            explanation_parts.append("• Monitor blood pressure and cholesterol")
        
        return "\n".join(explanation_parts)
    
    def identify_protective_factors(self, input_data: Dict[str, Any]) -> List[str]:
        """Identify protective factors from user input"""
        protective_factors = []
        
        # Good exercise capacity
        max_hr_expected = 220 - input_data.get('age', 50)
        if input_data.get('thalach', 150) > max_hr_expected * 0.85:
            protective_factors.append('Excellent exercise capacity')
        
        # Normal blood pressure
        if input_data.get('trestbps', 120) <= 120:
            protective_factors.append('Optimal blood pressure')
        
        # Good cholesterol
        if input_data.get('chol', 200) < 200:
            protective_factors.append('Healthy cholesterol levels')
        
        # No chest pain
        if input_data.get('cp', 0) == 4:  # Asymptomatic
            protective_factors.append('No chest pain symptoms')
        
        # No exercise-induced symptoms
        if input_data.get('exang', 0) == 0:
            protective_factors.append('No exercise-induced chest pain')
        
        # Normal ECG
        if input_data.get('restecg', 0) == 0:
            protective_factors.append('Normal resting ECG')
        
        return protective_factors
    
    def compare_with_population(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare user's risk relative to dataset statistics"""
        comparisons = {}
        
        # Age comparison
        user_age = input_data.get('age', 50)
        age_percentile = self.calculate_percentile(user_age, self.population_stats['age']['mean'], self.population_stats['age']['std'])
        comparisons['age'] = {
            'value': user_age,
            'percentile': age_percentile,
            'interpretation': f"Your age is at the {age_percentile:.0f}th percentile compared to the study population"
        }
        
        # Blood pressure comparison
        user_bp = input_data.get('trestbps', 120)
        bp_percentile = self.calculate_percentile(user_bp, self.population_stats['trestbps']['mean'], self.population_stats['trestbps']['std'])
        comparisons['blood_pressure'] = {
            'value': user_bp,
            'percentile': bp_percentile,
            'interpretation': f"Your blood pressure is at the {bp_percentile:.0f}th percentile"
        }
        
        # Cholesterol comparison
        user_chol = input_data.get('chol', 200)
        chol_percentile = self.calculate_percentile(user_chol, self.population_stats['chol']['mean'], self.population_stats['chol']['std'])
        comparisons['cholesterol'] = {
            'value': user_chol,
            'percentile': chol_percentile,
            'interpretation': f"Your cholesterol is at the {chol_percentile:.0f}th percentile"
        }
        
        return comparisons
    
    def calculate_percentile(self, value: float, mean: float, std: float) -> float:
        """Calculate percentile using normal distribution approximation"""
        try:
            from scipy import stats
            percentile = stats.norm.cdf(value, mean, std) * 100
            return max(0, min(100, percentile))
        except ImportError:
            # Fallback calculation if scipy not available
            z_score = (value - mean) / std if std > 0 else 0
            # Approximate percentile using z-score
            if z_score <= -2:
                return 2.5
            elif z_score <= -1:
                return 16
            elif z_score <= 0:
                return 50
            elif z_score <= 1:
                return 84
            elif z_score <= 2:
                return 97.5
            else:
                return 99
        except Exception:
            # Any other error, return 50th percentile
            return 50.0
    
    def get_feature_impact_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which features most influence the prediction"""
        if self.model is None:
            return {}
        
        try:
            # Get feature importance if available
            feature_importance = {}
            
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                importances = self.model.feature_importances_
                for i, feature in enumerate(self.feature_names):
                    if i < len(importances):
                        feature_importance[feature] = float(importances[i])
            
            elif hasattr(self.model, 'coef_'):
                # For linear models
                coefficients = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
                for i, feature in enumerate(self.feature_names):
                    if i < len(coefficients):
                        feature_importance[feature] = abs(float(coefficients[i]))
            
            # Normalize importance scores
            if feature_importance:
                max_importance = max(feature_importance.values())
                if max_importance > 0:
                    for feature in feature_importance:
                        feature_importance[feature] = feature_importance[feature] / max_importance
            
            # Create impact analysis
            impact_analysis = {}
            for feature, importance in feature_importance.items():
                user_value = input_data.get(feature, 0)
                impact_analysis[feature] = {
                    'importance': importance,
                    'user_value': user_value,
                    'impact_level': 'High' if importance > 0.7 else 'Medium' if importance > 0.3 else 'Low'
                }
            
            return impact_analysis
        
        except Exception as e:
            st.error(f"Error in feature impact analysis: {str(e)}")
            return {}


class PredictionHistory:
    """Manage prediction history tracking"""
    
    def __init__(self):
        self.history_file = "prediction_history.json"
        self.max_history = 50  # Maximum number of predictions to store
    
    def save_prediction(self, input_data: Dict[str, Any], prediction: int, confidence: float):
        """Save prediction to history"""
        try:
            # Load existing history
            history = self.load_history()
            
            # Create new prediction record
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'input_data': input_data,
                'prediction': prediction,
                'confidence': confidence
            }
            
            # Add to history
            history.append(prediction_record)
            
            # Keep only recent predictions
            if len(history) > self.max_history:
                history = history[-self.max_history:]
            
            # Save updated history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        
        except Exception as e:
            st.error(f"Error saving prediction history: {str(e)}")
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load prediction history"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Error loading prediction history: {str(e)}")
            return []
    
    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        history = self.load_history()
        return history[-limit:] if history else []
    
    def clear_history(self):
        """Clear prediction history"""
        try:
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
        except Exception as e:
            st.error(f"Error clearing history: {str(e)}")


def export_prediction_report(input_data: Dict[str, Any], prediction_result: Dict[str, Any], 
                           confidence_info: Dict[str, Any], risk_explanation: str) -> str:
    """Generate a downloadable prediction report"""
    
    report_lines = [
        "# Heart Disease Risk Assessment Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Patient Information",
        f"Age: {input_data.get('age', 'N/A')} years",
        f"Sex: {'Male' if input_data.get('sex', 0) == 1 else 'Female'}",
        "",
        "## Medical Parameters",
    ]
    
    # Add all input parameters
    parameter_labels = {
        'cp': 'Chest Pain Type',
        'trestbps': 'Resting Blood Pressure',
        'chol': 'Serum Cholesterol',
        'fbs': 'Fasting Blood Sugar',
        'restecg': 'Resting ECG',
        'thalach': 'Maximum Heart Rate',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'slope': 'ST Slope',
        'ca': 'Major Vessels',
        'thal': 'Thalassemia'
    }
    
    for param, label in parameter_labels.items():
        value = input_data.get(param, 'N/A')
        report_lines.append(f"{label}: {value}")
    
    report_lines.extend([
        "",
        "## Risk Assessment",
        f"**Risk Level:** {prediction_result.get('risk_category', 'Unknown')}",
        f"**Confidence:** {confidence_info.get('confidence_level', 'Unknown')} ({confidence_info.get('confidence_score', 0):.1%})",
        "",
        "## Detailed Analysis",
        risk_explanation,
        "",
        "## Important Disclaimer",
        "This assessment is for educational purposes only and should not replace professional medical advice.",
        "Always consult with qualified healthcare professionals for medical decisions.",
        "",
        "---",
        "Report generated by Heart Disease Risk Prediction System"
    ])
    
    return "\n".join(report_lines)