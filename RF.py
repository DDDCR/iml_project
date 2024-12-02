import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=4,
    min_samples_split=2,
    random_state=42
)

model.fit(X_scaled, y)

X_test = test_df[features]
X_test_scaled = scaler.transform(X_test)

test_predictions = model.predict(X_test_scaled)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'TARGET': test_predictions
})

# Ensure the TARGET column has appropriate precision
submission['TARGET'] = submission['TARGET'].round(9)

# Save to CSV
submission.to_csv('submission2.csv', index=False)
