# Importing the required packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import joblib

# Load the dataset
dataset = pd.read_csv("train.csv")

# Identify numerical and categorical columns
numerical_cols = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = dataset.select_dtypes(include=['object']).columns.tolist()

# Remove target and ID columns from categorical columns
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

# Fill missing values
for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

for col in numerical_cols:
    dataset[col].fillna(dataset[col].median(), inplace=True)

# Handle outliers
dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# Log transformation and domain processing
dataset['LoanAmount'] = np.log(dataset['LoanAmount'])
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome'])

# Drop unnecessary columns
dataset = dataset.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

# Label encoding categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

# Encode the target column separately
le_target = LabelEncoder()
dataset['Loan_Status'] = le_target.fit_transform(dataset['Loan_Status'])

# Train-test split
X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
y = dataset['Loan_Status']
RANDOM_SEED = 6

# RandomForest with GridSearchCV
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200, 400, 700],
    'max_depth': [10, 20, 30],
    'criterion': ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)
model_forest = grid_forest.fit(X, y)

# Save the model
joblib.dump(model_forest, 'RF_Loan_model.joblib')

# Load the model
loaded_model = joblib.load('RF_Loan_model.joblib')

# Test prediction with example data
data = [[
    1.0,    # Gender (Male)
    0.0,    # Married (No)
    0.0,    # Dependents
    0.0,    # Education (Graduate)
    0.0,    # Self_Employed (No)
    4.98745, # LoanAmount (log transformed)
    360.0,  # Loan_Amount_Term
    1.0,    # Credit_History (Outstanding Loan)
    2.0,    # Property_Area (Urban)
    8.698   # TotalIncome (log transformed)
]]

# Convert to DataFrame for prediction
data_df = pd.DataFrame(data, columns=X.columns)
prediction = loaded_model.predict(data_df)
prediction_label = le_target.inverse_transform(prediction)

print(f"Prediction is : {prediction_label[0]}")
