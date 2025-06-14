# Customer Churn Prediction using Random Forest and SMOTE

This project builds a machine learning model to predict customer churn using the Random Forest algorithm. It also addresses data imbalance using the SMOTE technique.

## 📂 Project Structure

Customer_Churn_Prediction/
├── customer_churn_dataset.csv
├── churn_model.py
└── README.md


## 📊 Dataset

- The dataset contains customer details and whether they have churned.
- Source: [Provide dataset link or origin if public]

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- RandomForestClassifier

## 📈 Model Workflow

1. Load and preprocess dataset
2. Encode categorical features
3. Split into train and test sets
4. Apply **SMOTE** to handle class imbalance
5. Train a **Random Forest** model
6. Evaluate using accuracy, confusion matrix, and classification report

## 📌 Installation

Install dependencies:

```bash
pip install pandas scikit-learn imbalanced-learn



Run the model:

python churn_model.py
📞 Results
Model performance is printed in the terminal.

Evaluation includes:

Accuracy Score

Confusion Matrix

Classification Report

✅ Future Improvements
Hyperparameter tuning

Feature engineering

Visualizations of model performance
