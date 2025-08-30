# Configuration file for Streamlit Heart Disease Prediction App

import os

# Application Configuration
APP_TITLE = "Heart Disease Risk Prediction"
APP_DESCRIPTION = "AI-powered heart disease risk assessment using machine learning"
VERSION = "1.0.0"

# Model Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "production", "complete_pipeline.pkl")
BACKUP_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "production", "final_model.pkl")

# Feature Configuration
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Feature Descriptions and Ranges
FEATURE_INFO = {
    'age': {
        'label': 'Age',
        'description': 'Age in years',
        'min_val': 20,
        'max_val': 100,
        'default': 50,
        'help': 'Enter your age in years (20-100)'
    },
    'sex': {
        'label': 'Sex',
        'description': 'Biological sex',
        'options': {'Female': 0, 'Male': 1},
        'default': 'Male',
        'help': 'Select your biological sex'
    },
    'cp': {
        'label': 'Chest Pain Type',
        'description': 'Type of chest pain experienced',
        'options': {
            'Typical Angina': 1,
            'Atypical Angina': 2,
            'Non-Anginal Pain': 3,
            'Asymptomatic': 4
        },
        'default': 'Asymptomatic',
        'help': 'Select the type of chest pain you experience'
    },
    'trestbps': {
        'label': 'Resting Blood Pressure',
        'description': 'Resting blood pressure in mm Hg',
        'min_val': 80,
        'max_val': 200,
        'default': 120,
        'help': 'Enter your resting blood pressure (80-200 mm Hg)'
    },
    'chol': {
        'label': 'Serum Cholesterol',
        'description': 'Serum cholesterol in mg/dl',
        'min_val': 100,
        'max_val': 600,
        'default': 200,
        'help': 'Enter your serum cholesterol level (100-600 mg/dl)'
    },
    'fbs': {
        'label': 'Fasting Blood Sugar',
        'description': 'Fasting blood sugar > 120 mg/dl',
        'options': {'No (≤120 mg/dl)': 0, 'Yes (>120 mg/dl)': 1},
        'default': 'No (≤120 mg/dl)',
        'help': 'Is your fasting blood sugar greater than 120 mg/dl?'
    },
    'restecg': {
        'label': 'Resting ECG',
        'description': 'Resting electrocardiographic results',
        'options': {
            'Normal': 0,
            'ST-T Wave Abnormality': 1,
            'Left Ventricular Hypertrophy': 2
        },
        'default': 'Normal',
        'help': 'Select your resting ECG results'
    },
    'thalach': {
        'label': 'Maximum Heart Rate',
        'description': 'Maximum heart rate achieved',
        'min_val': 60,
        'max_val': 220,
        'default': 150,
        'help': 'Enter your maximum heart rate achieved (60-220 bpm)'
    },
    'exang': {
        'label': 'Exercise Induced Angina',
        'description': 'Exercise induced angina',
        'options': {'No': 0, 'Yes': 1},
        'default': 'No',
        'help': 'Do you experience chest pain during exercise?'
    },
    'oldpeak': {
        'label': 'ST Depression',
        'description': 'ST depression induced by exercise relative to rest',
        'min_val': 0.0,
        'max_val': 6.0,
        'default': 1.0,
        'step': 0.1,
        'help': 'Enter ST depression value (0.0-6.0)'
    },
    'slope': {
        'label': 'ST Slope',
        'description': 'Slope of the peak exercise ST segment',
        'options': {
            'Upsloping': 1,
            'Flat': 2,
            'Downsloping': 3
        },
        'default': 'Flat',
        'help': 'Select the slope of your peak exercise ST segment'
    },
    'ca': {
        'label': 'Major Vessels',
        'description': 'Number of major vessels colored by fluoroscopy',
        'options': {'0': 0, '1': 1, '2': 2, '3': 3},
        'default': '0',
        'help': 'Number of major vessels (0-3) colored by fluoroscopy'
    },
    'thal': {
        'label': 'Thalassemia',
        'description': 'Thalassemia type',
        'options': {
            'Normal': 3,
            'Fixed Defect': 6,
            'Reversible Defect': 7
        },
        'default': 'Normal',
        'help': 'Select your thalassemia type'
    }
}

# UI Configuration
SIDEBAR_WIDTH = 300
MAIN_CONTENT_WIDTH = 700

# Page Configuration
PAGES = {
    'Prediction': '🔮 Heart Disease Prediction',
    'Data Explorer': '📊 Data Explorer',
    'Model Performance': '🤖 Model Performance',
    'Advanced Analysis': '🔬 Advanced Analysis',
    'Model Info': 'ℹ️ Model Information'
}

# Styling with improved accessibility
CUSTOM_CSS = """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        role: "banner";
    }
    
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
        role: "alert";
        aria-live: "polite";
    }
    
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    
    .feature-info {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Improve button accessibility */
    .stButton > button {
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stButton > button:focus {
        border: 2px solid #0066cc;
        outline: 2px solid #0066cc;
        outline-offset: 2px;
    }
    
    /* Improve form accessibility */
    .stSelectbox > div > div {
        border: 1px solid #ccc;
    }
    
    .stNumberInput > div > div > input {
        border: 1px solid #ccc;
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .low-risk {
            background-color: #ffffff;
            color: #000000;
            border: 3px solid #000000;
        }
        
        .high-risk {
            background-color: #ffffff;
            color: #000000;
            border: 3px solid #000000;
        }
    }
</style>
"""

# Risk Categories
RISK_CATEGORIES = {
    0: {
        'label': 'Low Risk',
        'color': 'green',
        'description': 'Based on the provided information, you have a low risk of heart disease.',
        'recommendations': [
            'Maintain a healthy lifestyle',
            'Regular exercise and balanced diet',
            'Regular health check-ups'
        ]
    },
    1: {
        'label': 'High Risk',
        'color': 'red',
        'description': 'Based on the provided information, you may have an elevated risk of heart disease.',
        'recommendations': [
            'Consult with a healthcare professional immediately',
            'Consider lifestyle modifications',
            'Follow up with regular cardiac monitoring'
        ]
    }
}