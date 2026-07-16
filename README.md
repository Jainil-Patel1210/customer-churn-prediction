# Customer Churn Prediction

An end-to-end machine learning project for predicting whether a telecom customer is likely to churn based on demographic information, subscribed services, contract details, and billing information.

The project covers the complete machine learning workflow, from exploratory data analysis and preprocessing to model training, hyperparameter tuning, and deployment through an interactive Streamlit application.

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Missing value handling
- One-hot encoding of categorical features
- Feature scaling
- Scikit-learn preprocessing pipelines
- Multiple model comparison
- Hyperparameter tuning using RandomizedSearchCV
- Model evaluation using multiple classification metrics
- Feature importance analysis
- Interactive churn prediction using Streamlit

## Dataset

The project uses the IBM Telco Customer Churn dataset containing customer information such as:

- Demographics
- Account tenure
- Phone and internet services
- Online security and technical support
- Contract type
- Payment method
- Monthly and total charges

The target variable is `Churn`, which indicates whether a customer discontinued the service.

## Machine Learning Pipeline

The following preprocessing steps were applied:

1. Converted `TotalCharges` from object to numerical format.
2. Handled missing values introduced during conversion.
3. Removed `customerID` as it is only an identifier.
4. Performed a stratified train-test split.
5. Applied median imputation and standardization to numerical features.
6. Applied most-frequent imputation and one-hot encoding to categorical features.
7. Combined preprocessing and model training using Scikit-learn pipelines.

Using a pipeline ensures that preprocessing transformations are learned only from the training data and are consistently applied during inference.

## Models Evaluated

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.804 | 0.648 | 0.572 | 0.608 | 0.836 |
| Random Forest | 0.760 | 0.541 | 0.647 | 0.590 | 0.815 |
| Gradient Boosting | 0.776 | 0.589 | 0.524 | 0.554 | 0.820 |
| Tuned Random Forest | 0.746 | 0.515 | 0.794 | 0.625 | 0.833 |

Logistic Regression achieved the highest test ROC-AUC and overall accuracy.

However, the tuned Random Forest was selected for deployment because it achieved a churn recall of approximately **79.4%**, allowing the model to identify a larger proportion of customers who actually churn.

In a customer retention scenario, identifying potential churners can be particularly valuable because the business can proactively target these customers with retention strategies.

The tuned Random Forest achieved a cross-validation ROC-AUC of approximately **0.846**.

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── 01_eda_and_preprocessing.ipynb
│
├── src/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application allows users to enter customer information and returns:

- A churn prediction
- The predicted probability of churn

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Future Improvements

- Add SHAP-based model explanations
- Improve feature engineering
- Experiment with XGBoost and LightGBM
- Add probability threshold optimization
- Add automated model training scripts
- Containerize the application using Docker
- Deploy the application to a cloud platform

## Author

Jainil Patel