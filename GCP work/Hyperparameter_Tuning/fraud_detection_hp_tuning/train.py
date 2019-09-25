import argparse
import datetime
from datetime import timedelta
from datetime import datetime as dt
import os
import pandas as pd
import numpy as np
import subprocess
import pickle
from google.cloud import storage
import hypertune
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.externals import joblib
from random import shuffle
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import recall_score
import category_encoders as ce
identity = 'train_identity.csv'
transaction = 'train_transaction.csv'

BUCKET_ID = 'frauddetectionkagglepkmatt'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--job-dir',  # handled automatically by AI Platform
    help='GCS location to write checkpoints and export models',
    required=True
)
parser.add_argument(
    '--max_depth',  # Specified in the config file
    help='Maximum depth of the XGBoost tree. default: 3',
    default=3,
    type=int
)
parser.add_argument(
    '--n_estimators',  # Specified in the config file
    help='Number of estimators to be created. default: 100',
    default=100,
    type=int
)
parser.add_argument(
    '--booster',  # Specified in the config file
    help='which booster to use: gbtree, gblinear or dart. default: gbtree',
    default='gbtree',
    type=str
)
parser.add_argument(
    '--learning_rate',  # Specified in the config file
    help='what learning_rate to use: 0.05 typical',
    default=0.05,
    type=float
)
parser.add_argument(
    '--bin_or_numerical_class',  # Specified in the config file
    help='whether to use binary or numerical label encoding for categoricals',
    default='numerical',
    type=str
)
parser.add_argument(
    '--extract_times',  # Specified in the config file
    help='whether to use feature engineer time features',
    default='true',
    type=str
)

args = parser.parse_args()
#  bucket holding the data
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


def load_and_merge_data(transaction_csv,identity_csv,isTrain):
    df_transaction = pd.read_csv(transaction_csv, index_col='TransactionID')
    df_identity = pd.read_csv(identity_csv, index_col='TransactionID')
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

def process_dates(df):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['TransactionDT_converted'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    df['time_year'] = df['TransactionDT_converted'].dt.year
    df['time_month'] = df['TransactionDT_converted'].dt.month
    df['time_dow'] = df['TransactionDT_converted'].dt.dayofweek
    df['time_hour'] = df['TransactionDT_converted'].dt.hour
    df['time_day'] = df['TransactionDT_converted'].dt.day
    df = df.drop(columns="TransactionDT_converted")
    df = df.drop(columns="time_year")
    df = df.drop(columns="time_month")
    df = df.drop(columns="time_day")
    return df

if args.extract_times == 'true':
    train = process_dates(train)

if args.bin_or_numerical_class == 'numerical':
    def numerically_encode_string_categoricals(df):
        for i in df.columns:
            if df[i].dtype == 'object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df[i].values) + list(df[i].values))
                df[i] = lbl.transform(list(df[i].values))
        return df
    train = numerically_encode_string_categoricals(train)
    #validate = numerically_encode_string_categoricals(validate)
    #Impute median for numerical and mode for categorical
    def impute_cat_and_num(df,numerical,categorical):
        fill_NaN_numerical = Imputer(missing_values=np.nan, strategy='median',axis=1)
        fill_NaN_categorical = Imputer(missing_values=np.nan, strategy='most_frequent',axis=1)
        df[numerical] = fill_NaN_numerical.fit_transform(df[numerical])
        df[categorical] = fill_NaN_categorical.fit_transform(df[categorical])
        return df
    train = impute_cat_and_num(train,numerical,categorical)

else:

    def binary_encode_categoricals(df,categorical):
        encoder = ce.BinaryEncoder(cols=categorical).fit(df)
        df = encoder.transform(df)
        return encoder,df
    encoder,train = binary_encode_categoricals(train,categorical)
    
    
    #Impute median for numerical and mode for categorical
    def impute_cat_and_num(df,numerical,categorical):
        fill_NaN_numerical = Imputer(missing_values=np.nan, strategy='median',axis=1)
        #fill_NaN_categorical = Imputer(missing_values=np.nan, strategy='most_frequent',axis=1)
        df[numerical] = fill_NaN_numerical.fit_transform(df[numerical])
        #df[categorical] = fill_NaN_categorical.fit_transform(df[categorical])
        return df
    train = impute_cat_and_num(train,numerical,categorical)

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

X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2,random_state=42)

# Create the regressor, here we will use a Lasso Regressor to demonstrate the use of HP Tuning.
# Here is where we set the variables used during HP Tuning from
# the parameters passed into the python script

#here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance


# clf = xgb.XGBClassifier(max_depth=args.max_depth,
#                              n_estimators=args.n_estimators,
#                              booster=args.booster,
#                              nthread=7,
#                              learning_rate=args.learning_rate
#                             )

clf = xgb.XGBClassifier(
        nthread=7,
        #n_estimators=1000,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        #learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='auto',
        booster=args.booster,
        random_state = 42,
        learning_rate=args.learning_rate
    )

#clf = xgb.XGBClassifier()

# Transform the features and fit them to the classifier
clf.fit(X_train, y_train)


# Calculate the mean accuracy on the given test data and labels.
#score = clf.score(X_test, y_test)

#calculate the recall score on test data and labels
#score = metrics.recall_score(y_test, clf.predict(X_test))

#roc score
score = metrics.roc_auc_score(y_test,clf.predict_proba(X_test)[:,1])

# The default name of the metric is training/hptuning/metric. 
# We recommend that you assign a custom name. The only functional difference is that 
# if you use a custom name, you must set the hyperparameterMetricTag value in the 
# HyperparameterSpec object in your job request to match your chosen name.
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
   hyperparameter_metric_tag='my_metric_tag',
   metric_value=score,
   global_step=1000)

# Export the model to a file
model_filename = 'model.pkl'
with open(model_filename, "wb") as f:
    pickle.dump(clf, f)

# Example: job_dir = 'gs://BUCKET_ID/xgboost_job_dir/1'
job_dir =  args.job_dir.replace('gs://', '')  # Remove the 'gs://'
# Get the Bucket Id
bucket_id = job_dir.split('/')[0]
# Get the path
bucket_path = job_dir[len('{}/'.format(bucket_id)):]  # Example: 'xgboost_job_dir/1'
# Upload the model to GCS
bucket = storage.Client().bucket(bucket_id)
blob = bucket.blob('{}/{}'.format(
    bucket_path,
    model_filename))

blob.upload_from_filename(model_filename)
