'''directories'''
DATA_DIR = "data"
PROCESSED_DIR = "processed"
FIG_DIR = "figures"

'''original data set is filtered by a particular value (`feature_values`) of 
`filter_by_feature` (or looped through a list of values) and weekly sales data averaged.
 must use a categorical feature, eg productgroup,gender,category,style

Below is configured to filter out data where productgroup==SHOES '''

filter_by_feature = 'productgroup'

feature_value = 'SHOES'


'''test/train data is split by time stamp.
 We will hold out a last portionof data to test our models on'''
train_size= 0.7

'''XGBoost hyperparameters '''
XGBparams = {"objective": "reg:linear",
              "booster" : "gbtree",
              "eta": 0.001,
              "max_depth": 2,
              "subsample": 0.7,
              "colsample_bytree": 0.7,
              "silent": 1,
              "thread": 1,
              "seed": 2017,
              }
XGBnum_boost_round = 5000

'''FUTURE FORECASTING PARAMETERs (turn these knobs)'''
'''create patterns for setting promo1 and promo2 features and see how it 
affects the sales forecast. Infinite possibilities- be creative!'''
'''below is an example of a simple oscillation where promo1 is 1 for even month
and 0 for odd months and promo2 is 1 for third quarters and 0 for other quarters'''
promo1_act_on = 'month'
promo2_act_on = 'quarter'
oscillate1 = (lambda x: 1 if x%2==0 else 0)
oscillate2 = (lambda x: 1 if x%2==0 else 0)

