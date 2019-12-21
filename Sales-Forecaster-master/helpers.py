import os
import settings
import numpy as np
import matplotlib.pyplot as plt


def rmse(y, yhat):
    #calculate root mean square error
    return np.sqrt(np.mean((yhat - y) ** 2))

def rmse_xg(yhat, y):
    #root mean square error function to be used to optimize XGBoost
    y = np.array(y.get_label())
    yhat = np.array(yhat)
    return "rmse", rmse(y,yhat)

def rmspe(y,yhat):
    #root mean square precentage error
    return np.sqrt(np.mean( ((y-yhat) / y)**2 ) )

def create_feature_map(features):
    #create feature importance map from XGBoost results
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()
    
def split(data, split_ratio):
    #split time series into train and test set, saving the end of the series
    #for the test set.
    X_train = data[:round(split_ratio*len(data))]
    X_valid = data[round(split_ratio*len(data)):]
    return X_train,X_valid

def save(filename):
    #save image as a png file
    savepath = os.path.join(settings.FIG_DIR, '{}.png'.format(filename))
    plt.savefig(savepath)
    plt.close()

if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    save("signal")
