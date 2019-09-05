import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import gc
import datetime
from datetime import timedelta
import re
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime as dt

class PreProcessData:

    # Load the 2 CSVs and left inner join the 2 dataframes on the TransactionID column:
    @staticmethod
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

    @staticmethod
    def process_dates(df):
        START_DATE = '2017-12-01'
        startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
        df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        df['time_year'] = df['TransactionDT'].dt.year
        df['time_month'] = df['TransactionDT'].dt.month
        df['time_dow'] = df['TransactionDT'].dt.dayofweek
        df['time_hour'] = df['TransactionDT'].dt.hour
        df['time_day'] = df['TransactionDT'].dt.day
        df = df.drop(columns="TransactionDT")
        df = df.drop(columns="time_year")
        df = df.drop(columns="time_month")
        return df

    #For each column check if there is any other column that is highly correlated. If so then remove the column that
    # has the most NAs out the pair.
    @staticmethod
    def remove_highly_correlated_columns(df, threshold, label):
        to_remove = []
        corr_matrix = df[df[label].notnull()].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        for i in upper.columns:
            for j in upper[i].index:
                if upper[i][j] > threshold:
                    if df[i].isna().sum() > df[j].isna().sum():
                        to_remove.append(i)
                    else:
                        to_remove.append(j)
        to_remove = list(set(to_remove))
        df = df.drop(columns=to_remove)
        return df

    #Create some groups of features by regex for some categorical variables:
    @staticmethod
    def process_device_names(df):
        id_30_regex = {'(?i)^.*Android.*$': 'Android',
                       '(?i)^.*iOS.*$': 'iOS',
                       '(?i)^.*Windows.*$': 'Windows',
                       '(?i)^.*Mac\sOS.*$': 'Mac'
                       }

        id_31_regex = {'(?i)^.*samsung.*$': 'samsung',
                       '(?i)^.*safari.*$': 'safari',
                       '(?i)^.*chrome.*$': 'chrome',
                       '(?i)^.*firefox.*$': 'firefox',
                       '(?i)^.*opera.*$': 'opera',
                       '(?i)^.*edge.*$': 'edge',
                       '(?i)^.*ie\s\d{1,2}.*$': 'ie',
                       '(?i)^.*android.*$': 'android',
                       '(?i)^.*google.*$': 'google',
                       }

        device_regex = {'^.*SM.*$|^.*(?i)SAMSUNG.*$': 'Samsung',
                        '(?i)^.*HUAWEI.*$|(?i)^hi\d.*$': 'Huawei',
                        '(?i)^.*TRIDENT.*$': 'Trident',
                        '(?i)rv\:.*$': 'Firefox',
                        '(?i)Moto.*$': 'Moto',
                        '(?i)Lg.*$': 'Lg',
                        '(?i)Linux.*$': 'Linux',
                        '(?i)HTC.*$': 'HTC',
                        '(?i)Hisense.*$': 'Hisense',
                        '(?i)Blade.*$': 'Blade',
                        '(?i)^.*XT\d+.*$': 'Motorola',
                        '(?i)^F\d{4}\sBuild.*$': 'Sony',
                        '(?i)^Lenovo.*$': 'Lenovo',
                        '(?i)^Redmi.*$': 'Xiaomi',
                        '(?i)^KFFOWI.*$': 'Amazon',
                        '(?i)^Pixel.*$|(?i)^Nexus.*$': 'Google',
                        '(?i)^Ilium.*$': 'Lanix',
                        '(?i)^Windows.*$': 'Windows',
                        '(?i)^\d{4}A.*$': 'Alcatel'}
        df['id_30'] = df['id_30'].replace(regex=id_30_regex)
        df['id_31'] = df['id_31'].replace(regex=id_31_regex)
        df['DeviceInfo'] = df['DeviceInfo'].replace(regex=device_regex)
        return df

    @staticmethod
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

    @staticmethod
    def impute_missing_values(df, numerical, categorical):
        for i in df[categorical]:
            df[i].fillna("NAN", inplace=True)
        fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
        df[numerical] = fill_NaN.fit_transform(df[numerical])
        return df

    @staticmethod
    def convert_numerical_categorical_to_strings(df,regex):
        for i in df:
            if re.match(regex, i) and df[i].dtype != 'O':
                df[i] = df[i].astype(str)
        return df

    @staticmethod
    def scale_numerical_fields(df, numerical):
        df[numerical] = StandardScaler().fit_transform(df[numerical])
        return df

    @staticmethod
    # reduce a group of numerical columns with PVA.  Also calculates new lists for
    # categorical and numerical
    def reduce_columns_with_PCA(df, regex, n_components,numerical,categorical):
        Vs = []
        for i in df:
            if re.match(regex, i):
                Vs.append(i)
        pca = PCA(n_components=n_components)
        PCA_V = pd.DataFrame(pca.fit_transform(df[Vs]), index=df.index).add_prefix('PCA_V_')
        df = df.drop(Vs, axis=1)
        df = df.merge(PCA_V, left_index=True, right_index=True)
        for i in Vs:
            numerical.remove(i)

        for i in df:
            if re.match('PCA_V.*', i):
                numerical.append(i)

        return df,numerical,categorical



    @staticmethod
    def assign_low_freq_values_as_other_in_column(df, colname, n):
        # df = df[colname]
        topn = df[colname].value_counts(dropna=False).head(n)
        if topn.shape[0] < n:
            return df
        uniques = df[colname].unique()
        others = []
        for i in uniques:
            if i not in topn:
                others.append(i)
        for i in others:
            df[colname].replace(i, 'other', inplace=True)
        return df

    @staticmethod
    def assign_low_freq_values_as_other_in_df(df, n, categorical):
        for i in df[categorical]:
            df = PreProcessData.assign_low_freq_values_as_other_in_column(df, i, 50)
        return df

    @staticmethod
    def one_hot_encode_and_merge_with_numerical(df,numerical,categorical,labels,isTrain):
        df_numerical = df[numerical]
        df_categorical_one_hot = pd.get_dummies(df[categorical])
        if isTrain:
            df = pd.concat([labels, df_numerical, df_categorical_one_hot], axis=1)
            del df_categorical_one_hot
            del df_numerical
            gc.collect()
        else:
            df = pd.concat([df_numerical, df_categorical_one_hot], axis=1)
            del df_categorical_one_hot
            del df_numerical
            gc.collect()
        return df

    @staticmethod
    def reduce_mem_usage(df):
        start_mem_usg = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
        NAlist = []  # Keeps track of columns that have missing values filled in.
        for col in df.columns:
            if df[col].dtype != object:  # Exclude strings

                # Print current column type
                #print("******************************")
                #print("Column: ", col)
                #print("dtype before: ", df[col].dtype)

                # make variables for Int, max and min
                IsInt = False
                mx = df[col].max()
                mn = df[col].min()

                # Integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(df[col]).all():
                    NAlist.append(col)
                    df[col].fillna(mn - 1, inplace=True)

                    # test if column can be converted to an integer
                asint = df[col].fillna(0).astype(np.int64)
                result = (df[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif mx < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif mx < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)

                            # Make float datatypes 32 bit
                else:
                    df[col] = df[col].astype(np.float32)

                # Print new column type
                #print("dtype after: ", df[col].dtype)
                #print("******************************")

        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024 ** 2
        print("Memory usage is: ", mem_usg, " MB")
        print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
        return df, NAlist




    @staticmethod
    def reconcile_features_from_test_to_train(df_test,path_to_model_feature_list):
        features_in_model = list(
            pd.read_pickle(path_to_model_feature_list)['name'])
        for i in df_test:
            if i not in features_in_model:
                df_test.pop(i)
        for i in features_in_model:
            if i not in df_test.columns:
                df_test[i]=0
        df_test.pop('isFraud')
        return df_test

    @staticmethod
    def pickle_df_and_columns(df, name, isTrain):
        # datetimestring = dt.now().strftime("%d-%m-%Y_%H-%M-%S")
        # os.mkdir('./data/processed/' + name)
        if isTrain:
            df.to_pickle('./data/processed/' + name + '.pkl')
            pd.DataFrame(list(df.columns), columns=['name']).to_pickle(
                './data/processed/' + name + '_feature_names.pkl')
        else:
            df.to_pickle('./data/processed/' + name + '.pkl')