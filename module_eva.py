import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Load the training and test datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#print(train_df.isnull().sum())
#train_df = train_df.drop('parentspecies', axis=1)

# data preprocess
features = ['MW', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumHBondDonors', 'NumOfConf']
X = train_df[features]
y = train_df['log_pSat_Pa']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

def data_visialization(train_df):
    sns.histplot(train_df['log_pSat_Pa'], kde=True)
    plt.title('Distribution of log_pSat_Pa')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(train_df.corr(), annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

def linear_reg(X_train, X_val, y_train, y_val):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)

    r2_lr = r2_score(y_val, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))

    return r2_lr, rmse_lr

def redge_reg(X_train, X_val, y_train, y_val):
    ridge = Ridge()
    parameters = {'alpha': [0.01, 0.1, 1, 10, 100]}
    ridge_reg = GridSearchCV(ridge, parameters, cv=5)
    ridge_reg.fit(X_train, y_train)
    y_pred_ridge = ridge_reg.predict(X_val)

    r2_ridge = r2_score(y_val, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))

    return r2_ridge, rmse_ridge

def rf(X_train, X_val, y_train, y_val):
    rf = RandomForestRegressor(random_state=42)
    parameters = {'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]}
    rf_reg = GridSearchCV(rf, parameters, cv=5)
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_val)

    r2_rf = r2_score(y_val, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
    
    return r2_rf, rmse_rf
    

if __name__ == '__main__':
    r2_lr, rmse_lr = linear_reg(X_train, X_val, y_train, y_val)
    print(f"Linear Regression R^2 Score: {r2_lr:.4f}")
    print(f"Linear Regression RMSE: {rmse_lr:.4f}")

    r2_ridge, rmse_ridge = redge_reg(X_train, X_val, y_train, y_val)
    print(f"Ridge Regression R^2 Score: {r2_ridge:.4f}")
    print(f"Ridge Regression RMSE: {rmse_ridge:.4f}")

    r2_rf, rmse_rf = rf(X_train, X_val, y_train, y_val)
    print(f"Random Forest R^2 Score: {r2_rf:.4f}")
    print(f"Random Forest RMSE: {rmse_rf:.4f}")


