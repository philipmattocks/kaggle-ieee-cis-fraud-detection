# [START setup]
import datetime
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import xgboost as xgb
from google.cloud import storage
import os
BUCKET_ID = 'frauddetectionkagglepkmatt'


# ---------------------------------------
# 1. Add code to download the data from GCS (in this case, using the publicly hosted data).
# AI Platform will then be able to use the data when training your model.
# ---------------------------------------
# [START download-data]


identity = 'train_identity.csv'
transaction = 'train_transaction.csv'

# Public bucket holding the census data
bucket = storage.Client().bucket(BUCKET_ID)

# Path to the data inside the public bucket
data_dir = 'data/raw/'

if not os.path.exists(identity):
    # Download the data
    blob = bucket.blob(''.join([data_dir, identity]))
    blob.download_to_filename(identity)
    
if not os.path.exists(transaction):    
    blob = bucket.blob(''.join([data_dir, transaction]))
    blob.download_to_filename(transaction)


def load_and_merge_data(transaction_csv,identity_csv,isTrain,nrows=1000000):
    df_transaction = pd.read_csv(transaction_csv, index_col='TransactionID',nrows=nrows)
    df_identity = pd.read_csv(identity_csv, index_col='TransactionID',nrows=nrows)
    df = pd.merge(df_transaction, df_identity, on='TransactionID', how='left')
    del df_transaction
    del df_identity
    if isTrain:
        labels = df[['isFraud']]
        df.pop('isFraud')
    else:
        labels = []
    return df, labels

train,labels  = load_and_merge_data(transaction,identity,isTrain=True)
#train,labels  = load_and_merge_data('gs://frauddetectionkagglepkmatt/data/raw/train_transaction.csv','gs://frauddetectionkagglepkmatt/data/raw/train_identity.csv',isTrain=True,nrows=5000)
# #validate,vallabels  = load_and_merge_data('./data/raw/test_transaction.csv','./data/raw/test_identity.csv',isTrain=False,nrows=5000)

#print(train.shape)

def get_lists_of_numerical_categorical(df,regex):
    #Regex for categorical fields:
    categorical = []
    numerical = []

    #Create lists of categorical and numeircal fields:
    for i in df:
        if re.match(regex, i):
            categorical.append(i)
        else:
            numerical.append(i)
    return numerical,categorical

cat_columns_regex='ProductCD|card[1-6]|addr\d|\w_emaildomain|M[1-9]|time_|Device\w+|id_12|id_13|id_14|id_15|id_16|id_17|id_18|id_19|id_20|id_21|id_22|id_23|id_24|id_25|id_26|id_27|id_28|id_29|id_30|id_31|id_32|id_33|id_34|id_35|id_36|id_37|id_38'
numerical,categorical = get_lists_of_numerical_categorical(train,cat_columns_regex)

def numerically_encode_string_categoricals(df):
    for i in df.columns:
        if df[i].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[i].values) + list(df[i].values))
            df[i] = lbl.transform(list(df[i].values))
    return df
train = numerically_encode_string_categoricals(train)
#validate = numerically_encode_string_categoricals(validate)

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA 
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

train = reduce_mem_usage(train)

def impute_cat_and_num(df,numerical,categorical):
    fill_NaN_numerical = Imputer(missing_values=np.nan, strategy='median',axis=1)
    fill_NaN_categorical = Imputer(missing_values=np.nan, strategy='most_frequent',axis=1)
    df[numerical] = fill_NaN_numerical.fit_transform(df[numerical])
    df[categorical] = fill_NaN_categorical.fit_transform(df[categorical])
    return df

train = impute_cat_and_num(train,numerical,categorical)

dtrain = xgb.DMatrix(train, labels)

param_dict = {
    'base_score':0.5,
    'booster':'gbtree',
    'colsample_bylevel':1,
    'colsample_bynode':1,
    'colsample_bytree':0.9,
    'gamma':0,
    'learning_rate':0.05,
    'max_delta_step':0,
    'max_depth':9,
    'min_child_weight':1,
    'n_estimators':1000,#this has to be entered explicitly in the function
    'n_jobs':1,
    'nthread':7,
    'objective':'binary:logistic',
    'random_state':42,
    'reg_alpha':0,
    'reg_lambda':1,
    'scale_pos_weight':1,
    'seed':42,
    'subsample':0.9,
    'tree_method':'auto',
    'verbosity':1
}
#param_dict = {}
clf = xgb.train(param_dict, dtrain, param_dict['n_estimators'])

model = 'model.bst'
clf.save_model(model)

# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_ID)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('fraud_detect_kaggle_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)
