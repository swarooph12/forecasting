# Sales Forecasting
------------------------------
This module predicts/forecasts sales numbers based on the company data (not provided here). The data contains time series sales history for a given product defined by an article ID number. The module groups these articles by some shared trait (shoes, tshirts, running, unisex, etc) and averages the data in a given retail week. It then uses this data to predict sales averages in future retail weeks. Some example results are plotted on `Figures` directory. The basic process is as follows:
  - Filter the dataset on some key value, i.e. take only 'SHOES' data, and average the weekly numbers
  - Calculate historical averages over relevant time frames and combine with other features such as promotional information to train/test forecasting models. Currently a Linear Regression model and a Extreme Gradient Boosted Decision Tree (XGBoost) can be used.
  - Use the models to predict future sales numbers. Adjust non-time related parameters, such as when promotions are occuring, to see how the sales predicitions are affected.
 
This module is intended to be modular and flexible.  Use `settings.py` to test any number of configurations!

### Settings
---
`settings.py` contains a list of relevant configuration options and a user should only have to modify this file to run different cases.

  - `DATA_DIR`,`PROCESSED_DIR` and `FIG_DIR` contain the paths for the original data, the processed data and the figures created by the module, respectively.
  - The data is filted by `filter_by_feature` keeping only data equal to the `feature_value`. For example, `filter_by_feature` can be "productgroup","category","gender","country". `feature_value` must match a specific value in the chosen `filter_by_feature`. Currently `settings.py` is configured to look at the data where the articles are 'SHOES'. 
  - `train_size` is the proportion of the data used for the training set. The rest of the data is held out to test and validate the trained models.
  - `XGBparams` and `XGBnum_boost_round` are hyperparameters for the XGBoost algorithm.
  - The final group of parameters are used to predict future sales averages. Think of these as the knobs that can be turned to affect futures sales. Currently, the "promo" features have this functionality.  Input some pattern and see how `predict.py`'s output changes.

### Set up
----
* Create directories for the raw data, processed data, and figures and update the variables in `settings.py` accordingly.  Store the raw data in `DATA_DIR`.
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 3.
    * For XGBoost you may need to follow installation directions [here](https://github.com/dmlc/xgboost/tree/master/python-package)
### Run
---
* Configure necessary parameters in `settings.py`
* Run `python assemble.py` to merge `sales.txt` with `article_attributes.txt`. This will create a new `sales.txt` file in `PROCESSED_DIR`.
* Run `python annotate.py` to filter and average the data and engineer the features which will be used to train the models. This will create a `*.txt` named according to `feature_value` in `settings.py` and a `features.txt` containing a list of features.
* Run `python predict.py`.  
    * This will print the root mean square precentage error of both the models (linear regression and XGBoost) calculated on the validiation set to the console. 
    * It will also create a "features importance" map resulting from the XGBoost model, saved in `FIG_DIR`.  This map ranks the features by importance to give one an idea of how each feature affects the predictions.  
    * Finally, the future is forecasted given the trained models, using the `promo1` and `promo2` patterns set in `settings.py`. A plot showing the historical data, the future data and the state of `promo1` and `promo2` is saved to `FIG_DIR`.
* As a bonus, run `python figloop.py`. This loops through all of the values of `filter_by_feature` training, testing and plotting the results of the model in a grid of plots. This will create `*_grid.png` file in `FIG_DIR`. 
### Future Works
---
* Add more non-time related features to train model and as more knobs to turn when forecasting the future.
* Multi level filtering - e.g. instead of looking at SHOES , just look at RUNNING SHOES.
* Can we forecast on the article level? Do we need more data for this?
* Improve models! Why can we predict SHOES better than TSHIRTS, for example? Is it a crucial feature we are missing? Do we need to use different algorithms? 





