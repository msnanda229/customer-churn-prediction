# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
churn_df = pd.read_csv(r'C:\Users\shanm\OneDrive\Documents\Customer_Churn_Prediction\customer_churn_dataset.csv')

# Strip any extra spaces in column names
churn_df.columns = churn_df.columns.str.strip()

# Drop non-numeric or irrelevant columns (like customer ID)
if 'customerID' in churn_df.columns:
    churn_df = churn_df.drop('customerID', axis=1)

# Ensure 'Churn' column is cleaned (if it's categorical like "Yes"/"No", convert to 0/1)
if churn_df['Churn'].dtype == 'object':
    churn_df['Churn'] = churn_df['Churn'].map({'Yes': 1, 'No': 0})

# Convert categorical variables to numeric using one-hot encoding
churn_df = pd.get_dummies(churn_df, drop_first=True)

# Split features and target
X = churn_df.drop('Churn', axis=1)
y = churn_df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE to balance the classes in training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

