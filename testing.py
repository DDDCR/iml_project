import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

df_train = train_data.copy(deep=True)
df_test = test_data.copy(deep=True)

df_train['parentspecies'] = df_train['parentspecies'].astype(str)

if not df_train['parentspecies'].isnull().all(): 
    most_common_category = df_train['parentspecies'].mode()[0] 
else:
    most_common_category = 'Unknown'


df_train['parentspecies'].fillna(most_common_category, inplace=True)

df_train.fillna(df_train.mean(numeric_only=True), inplace=True)

df_test['parentspecies'] = df_test['parentspecies'].astype(str)

if not df_test['parentspecies'].isnull().all():  
    most_common_category = df_test['parentspecies'].mode()[0] 
else:
    most_common_category = 'Unknown' 

df_test['parentspecies'].fillna(most_common_category, inplace=True)

df_test.fillna(df_test.mean(numeric_only=True), inplace=True)

df_train['log_pSat_Pa'] = np.where(df_train['log_pSat_Pa'] <= 0, np.nan, df_train['log_pSat_Pa'])
df_train['log_pSat_Pa'] = np.log10(df_train['log_pSat_Pa']).fillna(df_train['log_pSat_Pa'].mean())

df_train = df_train[df_train['parentspecies'].isin(['apin', 'toluene', 'decane'])]  
merged = pd.concat([df_train, df_test])
most_common_category = merged['parentspecies'].mode()[0]

mask = df_test['parentspecies'].isin(['apin', 'toluene', 'decane'])
df_test.loc[~mask, 'parentspecies'] = most_common_category

one_hot_train = pd.get_dummies(df_train['parentspecies'], prefix='parentspecies')
one_hot_test = pd.get_dummies(df_test['parentspecies'], prefix='parentspecies')

one_hot_train, one_hot_test = one_hot_train.align(one_hot_test, join='outer', axis=1, fill_value=0)

df_train = df_train.drop('parentspecies', axis=1).join(one_hot_train)
df_test = df_test.drop('parentspecies', axis=1).join(one_hot_test)

df_train = df_train.drop('ID', axis=1)
df_test = df_test.drop('ID', axis=1)

X = df_train.drop('log_pSat_Pa', axis=1)
y = df_train['log_pSat_Pa']

# Model training using K-Fold
result_model = {}
kf = KFold(n_splits=10)

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    lr = LinearRegression().fit(X_train_poly, y_train)

    X_val_poly = poly.transform(X_val)
    r2 = r2_score(y_val, lr.predict(X_val_poly))
    result_model[r2] = lr

best_model = result_model[max(result_model.keys())]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
best_model = best_model.fit(X_poly, y)

X_test_poly = poly.transform(df_test)
y_pred = best_model.predict(X_test_poly)

submission = pd.DataFrame({
    'Id': test_data['ID'],
    'target': y_pred
})

submission.to_csv('team25_y.csv', index=False)