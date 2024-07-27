import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

bike_df = pd.read_csv('./bike-sharing-demand/train.csv')
bike_df

bike_df.info()

bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])

bike_df['year'] = bike_df['datetime'].apply(lambda x: x.year)
bike_df['mbonth'] = bike_df['datetime'].apply(lambda x: x.month)
bike_df['day'] = bike_df['datetime'].apply(lambda x: x.day)
bike_df['hour'] = bike_df['datetime'].apply(lambda x: x.hour)

bike_df_new = bike_df.drop(['datetime','casual','registered'],axis=1)

bike_df_new['count'].hist()

def rmsle(y_test,pred):
    log_y = np.log1p(y_test)
    log_pred = np.log1p(pred)
    squared_log_error1 = (log_y-log_pred)**2
    mean_squared_log_error1 = np.mean(squared_log_error1)
    rmsle_result = np.sqrt(mean_squared_log_error1)

    return rmsle_result
def rmse(y_test, pred):
    rmse_result = np.sqrt(mean_squared_error(y_test,pred))
    return rmse_result

def get_eval_index(y_test,pred):
    rmsle_eval = rmsle(y_test,pred)
    rmse_eval = rmse(y_test,pred)

    mae_eval = mean_absolute_error(y_test,pred)
    print('RMSLE:{0:.4f}, RMSE:{1:.4f}, MAE:{2:.4f}'.format(rmsle_eval,rmse_eval, mae_eval))


y_target = bike_df_new['count']
X_ftrs = bike_df_new.drop(['count'], axis=1)

xtrain, xval, ytrain, yval = train_test_split(X_ftrs, y_target, test_size=0.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(xtrain,ytrain)
pred_lr_reg = lr_reg.predict(xval)
get_eval_index(yval, pred_lr_reg)

check_df = pd.DataFrame(yval.values, columns=['actual_y'])
check_df['pred_y'] = pred_lr_reg
check_df['diff'] = np.abs(check_df['pred_y']-check_df['actual_y'])
check_df.sort_values(by='diff', ascending=False).reset_index()[:10]


yval.hist()


log1p_yval = np.log1p(yval)
log1p_yval.hist()


y_target_log1p = np.log1p(y_target)


y_target_log1p = np.log1p(y_target)
X_ftrs = bike_df_new.drop(['count'], axis=1)

xtrain, xval, ytrain_log, yval_log = train_test_split(X_ftrs, y_target_log1p,test_size=0.3, random_state=0)

lr_reg = LinearRegression()
lr_reg.fit(xtrain,ytrain_log)
pred_lr1_reg = lr_reg.predict(xval)

yval_exp = np.expm1(yval_log)

pred_lr1_exp = np.expm1(pred_lr1_reg)

get_eval_index(yval_exp, pred_lr1_exp)

coef = pd.Series(lr_reg.coef_, index=X_ftrs.columns)
coef_sorted = coef.sort_values(ascending=False)
sns.barplot(x=coef_sorted.values, y=coef_sorted.index)


bike_df_new['year'].value_counts()


X_ftrs.info()

['year','month','day','hour','holiday', 'workingday','season','weather']



X_ftrs_oh = pd.get_dummies(X_ftrs, columns=['year','month','day','hour','holiday', 'workingday','season','weather'])

X_ftrs_oh

y_target_log1p = np.log1p(y_target)

X_ftrs_oh

y_target_log1p = np.log1p(y_target)

xtrain2, xval2, ytrain2, yval2 = train_test_split(X_ftrs_oh, y_target_log1p, test_size=0.3, random_state=0)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

lr_reg = LinearRegression()
ridge = Ridge(alpha= 10)
lasso = Lasso(alpha= 0.001)

models = [lr_reg, ridge, lasso]

for model in models:
    model.fit(xtrain2,ytrain2)
    pred= model.predict(xval2)
    yval_exp1 = np.expm1(yval2)
    pred_exp1 = np.expm1(pred)
    print('\n ###', model.__class__.__name__, '### \n')
    get_eval_index(yval_exp1, pred_exp1)


coef = pd.Series(ridge.coef_, index=X_ftrs_oh.columns)
coef_sorted_r = coef.sort_values(ascending=False)[:11]
coef_sorted_r

sns.barplot(x=coef_sorted_r.values, y= coef_sorted_r.index)

coef_l = pd.Series(lasso.coef_, index=X_ftrs_oh.columns)
coef_sorted_l = coef.sort_values(ascending=False)[:11]
coef_sorted_l

sns.barplot(x=coef_sorted_l.values, y= coef_sorted_l.index)

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=500)

rf_reg.fit(xtrain2,ytrain2)
pred_rf = rf_reg.predict(xval2)

yval_expm1_rf = np.expm1(yval2)

pred_expm1_rf = np.expm1(pred_rf)

get_eval_index(yval_expm1_rf,pred_expm1_rf)
