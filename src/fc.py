import pandas as pd
import numpy as np

def column_convert_date(df):
    '''automatically tries to select columns that looks like dates and converts them'''
    df = df.apply(lambda col: pd.to_datetime(col, errors='ignore')
          if col.dtypes == object
          else col,
          axis=0)
    return

def add_date_columns(df, column_name=None, date_desc=None):
    if column_name is None or len(column_name) == 0:
        column_name = df_num.select_dtypes(include=[np.datetime64]).columns
    if data_desc is None or len(data_desc)==0:
        data_desc = ['year','month','day','dayofweek','weekday_name','hour','time','minute']
    for dt_type in date_desc:
        df[column_name + '_' + dt_type] = pd.DatetimeIndex(df[column_name]).year

    if 1 ==2:
        df[column_name + '_year'] = pd.DatetimeIndex(df[column_name]).year
        df[column_name + '_month'] = pd.DatetimeIndex(df[column_name]).month
        df[column_name + '_day'] = pd.DatetimeIndex(df[column_name]).day
        df[column_name + '_hour'] = pd.DatetimeIndex(df[column_name]).hour
        df[column_name + '_dayofweek'] = pd.DatetimeIndex(df[column_name]).dayofweek
        df[column_name + '_weekday'] = pd.DatetimeIndex(df[column_name]).weekday_name
        df[column_name + '_time'] = pd.DatetimeIndex(df[column_name]).time
        df[column_name + '_minute'] = pd.DatetimeIndex(df[column_name]).minute
    return



df = pd.read_csv('data/churn_train.csv')
column_convert_date(df)
