# Heart Disease ML Pipeline

A comprehensive machine learning pipeline for analyzing, predicting, and visualizing heart disease risks using the UCI Heart Disease dataset.

## Project Overview

This project implements an end-to-end machine learning workflow that includes:

- **Data Preprocessing**: Cleaning, encoding, and scaling of the UCI Heart Disease dataset
- **Feature Engineering**: PCA analysis and feature selection techniques
- **Supervised Learning**: Multiple classification models (Logistic Regression, Decision Tree, Random Forest, SVM)
- **Unsupervised Learning**: K-Means and Hierarchical clustering analysis
- **Model Optimization**: Hyperparameter tuning using GridSearch and RandomSearch
- **Web Interface**: Interactive Streamlit application for real-time predictions
- **Deployment**: Public access via Ngrok integration

## Dataset

The project uses the UCI Heart Disease dataset (Cleveland database) with 303 instances and 14 features:

- **age**: Age in years
- **sex**: Sex (1 = male, 0 = female)
- **cp**: Chest pain type (1-4)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (1-3)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
- **target**: Heart disease presence (0 = no, 1-4 = yes, degrees of severity)

## Project Structure

```
├── data/
│   ├── raw/                    # Original dataset files
│   └── processed/              # Cleaned and processed data
├── notebooks/                  # Jupyter notebooks for analysis
├── src/                        # Source code modules
├── models/                     # Trained model files
│   ├── supervised/
│   ├── unsupervised/
│   ├── optimized/
│   └── production/
├── ui/                         # Streamlit web interface
├── results/                    # Analysis results and visualizations
│   ├── eda/
│   ├── pca/
│   ├── feature_selection/
│   ├── model_evaluation/
│   ├── clustering/
│   └── hyperparameter_tuning/
├── tests/                      # Unit and integration tests
├── deployment/                 # Deployment scripts and configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd heart-disease-ml-pipeline
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, matplotlib, seaborn, streamlit; print('All dependencies installed successfully!')"
   ```

## Usage

### Running the Analysis Pipeline

1. **Data Preprocessing**:
   ```bash
   python src/data_processor.py
   ```

2. **Feature Engineering**:
   ```bash
   python src/feature_engineering.py
   ```

3. **Model Training**:
   ```bash
   python src/model_trainer.py
   ```

4. **Model Evaluation**:
   ```bash
   python src/model_evaluator.py
   ```

### Running the Web Interface

1. **Start the Streamlit application**:
   ```bash
   streamlit run ui/streamlit_app.py
   ```

2. **Access the application**:
   - Local: http://localhost:8501
   - Public (via Ngrok): Run deployment script for public URL

### Jupyter Notebooks

Explore the analysis step-by-step using the provided notebooks:

```bash
jupyter notebook notebooks/
```

## Features

### Machine Learning Models

- **Logistic Regression**: Linear classification with regularization
- **Decision Tree**: Tree-based classification with pruning
- **Random Forest**: Ensemble method with feature importance
- **Support Vector Machine**: Kernel-based classification

### Analysis Capabilities

- **Exploratory Data Analysis**: Statistical summaries and visualizations
- **Principal Component Analysis**: Dimensionality reduction and variance analysis
- **Feature Selection**: Multiple selection techniques (RFE, importance-based, statistical)
- **Clustering Analysis**: Pattern discovery in heart disease data
- **Hyperparameter Optimization**: Automated model tuning

### Web Interface Features

- **Real-time Predictions**: Input health parameters and get instant risk assessment
- **Interactive Visualizations**: Explore data patterns and model performance
- **Model Comparison**: Compare different algorithms and their performance
- **Risk Analysis**: Detailed explanation of prediction results

## Model Performance

The pipeline evaluates models using multiple metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- Cleveland Clinic Foundation for the original data collection
- Scikit-learn community for machine learning tools
- Streamlit team for the web framework

## Contact

For questions or suggestions, please open an issue in the repository.