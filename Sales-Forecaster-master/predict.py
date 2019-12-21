import os
import settings
from helpers import rmse,rmse_xg,rmspe,split,create_feature_map,save
import pandas as pd
import numpy as np
from sklearn import linear_model
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
import annotate
import datetime

import warnings
warnings.filterwarnings("ignore")

def read(filename = "filtered_processed_df"):
    #merged data set
    sales = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "sales.txt"), parse_dates=[6])
    #filters, processed data set
    data = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "{}.txt".format(filename)),
                       parse_dates=True,index_col=[0])
    #list of engineered features
    with open(os.path.join(settings.PROCESSED_DIR, "features.txt"), "r") as f:
        features = f.read().splitlines() 
    return sales, data, features

def train_test_LinearReg(X_train,X_valid,features):
    #train and test a simple linear regression model, reporting the root mean square precentage error
    model = linear_model.LinearRegression()
    model.fit(X_train[features], X_train.sales)
    X_valid['yhat']=model.predict(X_valid[features])    
    error = rmspe(X_valid.sales, X_valid.yhat)
    print('Linear Regression root mean square precentage error: ',
          error)
    X_out = pd.concat([X_train,X_valid])
    return X_out,X_valid.yhat,model,error

def train_test_XGboost(X_train,X_valid,features,params,num_boost_round,verbose = False):
    #train and test an extreme gradient boosted decision tree, outputting map of feature importance
    y_train = X_train.sales
    y_valid = X_valid.sales
    X_train_xgb = X_train[features]
    X_valid_xgb = X_valid[features]

    dtrain = xgb.DMatrix(X_train_xgb, y_train)
    dvalid = xgb.DMatrix(X_valid_xgb, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    
    # Training the tree:
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                early_stopping_rounds=500, feval=rmse_xg, verbose_eval=verbose)

    print("Validating")
    gbm_yhat = gbm.predict(xgb.DMatrix(X_valid_xgb))
    gbm_yhat_train = gbm.predict(xgb.DMatrix(X_train_xgb))

    error = rmspe(y_valid, gbm_yhat)
    print('XGBOOST root mean square precentage error: {:.6f}'.format(error))

    #feature importance map
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    fig_featp = featp.get_figure()
    #fig_featp.savefig(os.path.join(settings.FIG_DIR, 'feature_importance_xgb.png'))

    yhat_train = pd.DataFrame(data=gbm_yhat_train,index=y_train.index)
    yhat_valid = pd.DataFrame(data=gbm_yhat,index=y_valid.index)
    X_train.merge(yhat_train,left_index=True, right_index=True, how = 'right')
    X_valid.merge(yhat_valid,left_index=True, right_index=True, how = 'right')
    X_out = pd.concat([X_train,X_valid])

    return X_out,gbm,yhat_valid,error

def forecast_future(data,model,features,weeks_out = 52,use_XGBOOST = 0):
    '''build a dataframe with future dates and use results of model with some other made
    up features to predict future sales '''
    '''if using XGboost model, set use_XGBOOST to 1 '''
    #create dataframe for future sales
    last_date = list(data.index)[-1]
    projected_dates = [last_date + datetime.timedelta(weeks=x) for x in range(1, weeks_out)]
    projected_sales = pd.DataFrame(data=projected_dates,columns=['retailweek'])
    projected_sales.set_index('retailweek',inplace=True)
    #create data features, and fill in with historical sales averages
    projected_date_features = annotate.make_date_features(projected_sales)
    X_project,date_features=annotate.average_date_features(data,projected_sales,projected_date_features)
    X_project=X_project[last_date:]
    #make up some features
    #promo functions can be changed in settings.py to arbitrary values
    X_project['promo1'] = X_project[settings.promo1_act_on].apply(settings.oscillate1)
    X_project['promo2'] = X_project[settings.promo2_act_on].apply(settings.oscillate2)
    #use average ratio,TODO:MAKE MORE INTERSTING FUNCTION
    X_project['ratio'] = sales.groupby('retailweek').ratio.mean().mean() 
    #make predictions
    if use_XGBOOST:
        X_project['predictions'] = model.predict(xgb.DMatrix(X_project[features]))
    else:
        X_project['predictions'] = model.predict(X_project[features])
    #plot the results
    plt.figure(figsize=(20,10))
    plt.xticks(rotation=70)
    plt.plot(data.sales,label="historical sales data")
    plt.plot(X_project.predictions,label="future sales projections")
    plt.plot(40*X_project.promo1+50,label="promo1 state")
    plt.plot(40*X_project.promo2+50,label="promo2 state")
    plt.title("projected sales")
    plt.ylabel("Sales")
    plt.legend(loc='best')
    save("projections_"+str(settings.feature_value))
    
    return X_project


if __name__ == "__main__":
    sales, filtered_data,features  = read(filename=str(settings.feature_value))
    X_train,X_valid = split(filtered_data,settings.train_size)
    X_out,gbm,yhat_valid,error=train_test_XGboost(X_train,X_valid,features,
                                                  settings.XGBparams,settings.XGBnum_boost_round,
                                                  verbose = False)
    save('feature_importance_'+settings.feature_value)
    X_out,yhat,linreg,error = train_test_LinearReg(X_train,X_valid,features)
    X_project = forecast_future(filtered_data,gbm,features,use_XGBOOST=1)




