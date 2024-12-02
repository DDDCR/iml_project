import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint as sp_randint

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df = train_df.dropna()
numeric_cols = test_df.select_dtypes(include=[np.number]).columns
categorical_cols = test_df.select_dtypes(include=[object]).columns

test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].mean())
test_df[categorical_cols] = test_df[categorical_cols].fillna(test_df[categorical_cols].mode().iloc[0])

assert not train_df.isnull().values.any(), "Training data contains NaN values"
assert not test_df.isnull().values.any(), "Test data contains NaN values"

features = ['MW', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumOfN', 'NumHBondDonors', 'NumOfConf', 'NumOfConfUsed', 'C=C (non-aromatic)', 'C=C-C=O in non-aromatic ring', 'hydroxyl (alkyl)', 'aldehyde', 'ketone', 'carboxylic acid', 'ester', 'ether (alicyclic)', 'nitrate', 'nitro', 'aromatic hydroxyl', 'carbonylperoxynitrate', 'peroxide', 'hydroperoxide', 'carbonylperoxyacid', 'nitroester']
categorical_features = ['parentspecies']
target = 'log_pSat_Pa'

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

X = train_df[features + categorical_features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'regressor__n_estimators': sp_randint(50, 200),
    'regressor__max_depth': sp_randint(3, 10),
    'regressor__min_samples_split': sp_randint(2, 11),
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2]
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=3, scoring='r2', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_val)
print(f'R2 Score: {r2_score(y_val, y_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_val, y_pred)}')

X_test = test_df[features + categorical_features]
test_predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    'ID': test_df['ID'],
    'TARGET': test_predictions
})

submission['TARGET'] = submission['TARGET'].round(9)
submission.to_csv('submission.csv', index=False)