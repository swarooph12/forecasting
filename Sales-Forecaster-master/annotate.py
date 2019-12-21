import os
import settings
import pandas as pd
import numpy as np
import helpers

import warnings
warnings.filterwarnings("ignore")

def read():
    '''read in merged data file'''
    data = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "sales.txt"), parse_dates=[6])
    return data

def avg_by_feature(data,feature,feature_value):
    '''filter data on a specific feature value (set in settings.py) and average sales in a given week'''
    feat_avg = data[data[feature] == feature_value]
    feat_avg = pd.DataFrame(feat_avg.groupby('retailweek').sales.mean())
    return feat_avg

def make_date_features(data):
    '''create date features from retailweek index'''
    data.reset_index(inplace=True)
    data.set_index('retailweek',inplace=True)
    date_features = ['quarter','month','weekofyear']
    data['quarter'] = data.index.quarter
    data['month'] = data.index.month
    data['weekofyear'] = data.index.weekofyear
    return date_features

def average_date_features(X_train,X_valid,date_features):
    '''create features based on historical sales (training set). Input data needs to be 
    pre-split so that the average is only taken over the training (historical) data set'''

    #use the overall average sales of the training set as a feature in the test set 
    overall_avgs = X_train['sales'].mean()
    X_train['overall_avg'] = overall_avgs
    X_valid['overall_avg'] = overall_avgs

    date_avg_features = ['overall_avg'] #list to store new features
    #calculate averages for the date features from the training set, to be used in the test set
    for item in date_features:
        item_avg = X_train.groupby([item]).sales.mean()
        new_feature_name = item + '_past_average'
        date_avg_features.append(new_feature_name)
        X_train[new_feature_name] = X_train[item].apply(lambda x: item_avg[x])
        X_valid[new_feature_name] = X_valid[item].apply(lambda x: item_avg[x])
    #recombine train and valid set for simplicity
    X = pd.concat([X_train,X_valid])
    return X, date_avg_features

def create_features(data, X,date_avg_features):
    '''create the rest of the features to use in training the model'''
    features = date_avg_features
    #note: more features can be added here but they but must already be in numerical form
    #TODO: add ability to add more features including mapping of non-numeric features
    features.extend(['promo1','promo2','ratio'])  
    #avg from the original dataset
    for item in ['promo1','promo2','ratio']:
        item_avg = pd.DataFrame(data.groupby('retailweek')[item].mean())
        X = pd.concat([X,item_avg],axis=1)
    
    return X, features

def feature_loop_script(data,filter_feature,feature_value):
    #script for whole feature creating process
    feature_value = avg_by_feature(data,filter_feature,feature_value)
    date_features = make_date_features(feature_value)
    X_train,X_valid = helpers.split(feature_value,settings.train_size)
    X_historical,date_avg_features = average_date_features(X_train,X_valid,date_features)
    X_final,features_final = create_features(data,X_historical,date_avg_features)
    return X_final,features_final

def write(data,features,filename = "filtered_processed_df"):
    #save the filtered data with new engineered features along with a list of the filters
    #for later us
    data.to_csv(os.path.join(settings.PROCESSED_DIR, "{}.txt".format(filename)), index=True)
    with open(os.path.join(settings.PROCESSED_DIR, "features.txt"), "w") as fh:
        for item in features:
            fh.write("%s\n" % item)
   

if __name__ == "__main__":
    sales = read()
    X_final,features_final = feature_loop_script(sales,settings.filter_by_feature,settings.feature_value)
    write(X_final,features_final,filename = str(settings.feature_value))
