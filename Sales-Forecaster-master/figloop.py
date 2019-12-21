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
import predict
import math

def TestingModel_PlotGrid(model = "linear_regression"):
    '''choose either `linear_regression` or `XGB` TODO: make more general '''
    '''loop for the different values of a chosen feature, training, testing and plotting 
    the results of each one in a nice grid'''
    sales = annotate.read()
    #get a list of the unique values for the feature
    feature = settings.filter_by_feature
    feature_list = list(sales[feature].unique())
    fig, axs = plt.subplots(math.ceil(len(feature_list)/2),2,figsize=(15,15),sharex='col', sharey='row')
    fig.autofmt_xdate()
    #plt.xticks(rotation=70)
    axs = axs.ravel()
    for i,item in enumerate(feature_list):
        print(item)
        X_final,features_final= annotate.feature_loop_script(sales,feature,item)
        X_train_final,X_valid_final = split(X_final,.7)
        if model == "linear_regression":
            X_LinRegOut,predictions, model_linreg, error = predict.train_test_LinearReg(X_train_final,X_valid_final,
                                                                                        features_final)
        elif model == "XGB":
            X_out,gbm,predictions,error = predict.train_test_XGboost(X_train_final,X_valid_final,features_final
                                                                     ,settings.XGBparams,settings.XGBnum_boost_round)
            save('feature_importance_'+str(item))

        else: print("not a valid model"); break
        
        axs[i].plot(X_train_final['sales'],label="train")
        axs[i].plot(X_valid_final['sales'],label="validate")
        axs[i].plot(predictions,label='predictions')
        axs[i].legend(loc='best')
        axs[i].set_title(item+" rmse: "+str(error))
    fig.text(0.09, 0.55, 'Sales',fontsize=20, ha='center', va='center', rotation='vertical')

    save(feature+"_using_"+model+"_grid")

    
if __name__ == "__main__":
    TestingModel_PlotGrid(model='XGB')
