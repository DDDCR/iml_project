# iml_project

## data.py
preprocess and visialization the train dataset,
find features, ['MW', 'NumOfAtoms', 'NumOfC', 'NumOfO', 'NumHBondDonors', 'NumOfConf'], are related with log_pSat_Pa

## module_eva.py
evaluate the model of linear regression, redge regression, and random forrest with r2 and rmse.
result:
Linear Regression R^2 Score: 0.6372
Linear Regression RMSE: 1.8809
Ridge Regression R^2 Score: 0.6372
Ridge Regression RMSE: 1.8809
Random Forest R^2 Score: 0.6603
Random Forest RMSE: 1.8201

## RF.py
the result saved in submission.csv file
kaggle score = 0.6753
