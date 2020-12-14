import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing_df = pd.read_csv("Data.csv")
#housing_df.info()
#housing_df.hist(bins=50,figsize=(20,15))
'''
def split_train_test_model(data,test_ratio):
    np.random.seed(1)
    shuffled_data=np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices= shuffled_data[:test_set_size]
    train_indices=shuffled_data[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set, test_set = split_train_test_model(housing_df,test_ratio=0.2)
print(len(train_set),len(test_set))

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
train_set, test_split = train_test_split(housing_df,test_size=0.2,random_state=1)
'''
from sklearn.model_selection import StratifiedShuffleSplit
split_data = StratifiedShuffleSplit(n_splits=1,random_state=1,test_size=0.2)
for train_index, test_index in split_data.split(housing_df,housing_df['CHAS']):
    strat_train_set= housing_df.loc[train_index]
    strat_test_set= housing_df.loc[test_index]
    
#strat_train_set['CHAS'].value_counts()
housing_dfN=strat_train_set.copy()
corr_matrix = housing_dfN.corr()

#corr_matrix['MEDV'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes_housing=['MEDV','ZN','LSTAT','RM']
#scatter_matrix(housing_df[attributes_housing],figsize=(12,8))
#housing_df.plot(kind="scatter",x='RM',y='MEDV',alpha=0.8)
#housing_df['TAXRM']=housing_df['TAX']/housing_df['RM']
#to fill empty data
#median_fill = housing_df['RM'].median
#housing_df['RM'].fillna(median_fill)
housing_dfN= strat_train_set.drop('MEDV',axis=1)
housing_dfN_labels= strat_train_set['MEDV'].copy()
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy='median')
imputer.fit(housing_dfN)
X= imputer.transform(housing_dfN)
housing_tr = pd.DataFrame(X,columns= housing_dfN.columns)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline= Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('stdScalar',StandardScaler())
        ])
housing_num_tr=my_pipeline.fit_transform(housing_tr)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
regressor = LinearRegression()
regressorD = DecisionTreeRegressor()
regressorRF= RandomForestRegressor()

regressor.fit(housing_num_tr,housing_dfN_labels)
regressorD.fit(housing_num_tr,housing_dfN_labels)
regressorRF.fit(housing_num_tr,housing_dfN_labels)
#prepared_data = my_pipeline.transform(housing_dfN)
#regressor.predict(prepared_data)
from sklearn.metrics import mean_squared_error
housing_predictions= regressor.predict(housing_num_tr)
#housing_predictions= regressorD.predict(housing_num_tr)
lin_mse= mean_squared_error(housing_dfN_labels,housing_predictions)
lin_rmse= np.sqrt(lin_mse)

from sklearn.model_selection import cross_val_score
cross_validated_scoresD = cross_val_score(regressorD,housing_num_tr,housing_dfN_labels,scoring='neg_mean_squared_error',cv=10)
rmse_scoresD= np.sqrt(-cross_validated_scoresD)
cross_validated_scoresR = cross_val_score(regressor,housing_num_tr,housing_dfN_labels,scoring= 'neg_mean_squared_error',cv=10) 
rmse_scoresR = np.sqrt(-cross_validated_scoresR)
cross_validated_scoresRF = cross_val_score(regressorRF,housing_num_tr,housing_dfN_labels,scoring= 'neg_mean_squared_error',cv=10) 
rmse_scoresRF = np.sqrt(-cross_validated_scoresRF)

def display_scores(scores):
    print('Scores',scores)
    print('Mean',np.mean(scores))
    print('Standard Deviation',np.std(scores))

#display_scores(rmse_scoresD)
#display_scores(rmse_scoresR)
#display_scores(rmse_scoresRF)


from joblib import dump,load
dump(regressorRF,'RealEstate.joblib')
X_test= strat_test_set.drop("MEDV",axis=1)
y_test = strat_test_set["MEDV"].copy()
X_test_data = my_pipeline.transform(X_test)
pred= regressorRF.predict(X_test_data)
ran_mse_test = mean_squared_error(y_test,pred)
ran_rmse_test= np.sqrt(ran_mse_test)
print(pred,list(y_test))