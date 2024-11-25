import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load the training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# data preprocess
features = ['MW', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumHBondDonors', 'NumOfConf']
X = train_df[features]
y = train_df['log_pSat_Pa']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)
parameters = {'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]}
rf_reg = GridSearchCV(rf, parameters, cv=5)
rf_reg.fit(X_train, y_train)

X_test = test_df[features]
X_test_scaled = scaler.transform(X_test)

test_predictions = rf_reg.predict(X_test_scaled)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'TARGET': test_predictions
})

# Ensure the TARGET column has appropriate precision
submission['TARGET'] = submission['TARGET'].round(9)

# Save to CSV
submission.to_csv('submission.csv', index=False)
