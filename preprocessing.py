import pandas as pd
import numpy as np
from datetime import datetime



def load_csv(csv_name):
    data = pd.read_csv(f'data/{csv_name}.csv', encoding='euc-kr')
    return data

def create_nan_data(df):
    data = df.replace('-',np.NaN)
    return data

def delete_nan_data(df):
    data = df
    #data = df.fillna(method="bfill")
    return data

def join_data(df1,df2,dataName):
    data = pd.concat([df1,df2[f'{dataName}']],axis=1)
    return data

def join_data_2(df1,df2):
    data = pd.merge(df1,df2,on='시점',how='left')
    return data

def save_df_to_csv(df,fileName):
    path = 'C:/Users/park/Documents/GitHub/biz_predict/data/'
    df.to_csv(f'{path}{fileName}.csv',encoding='euc-kr',mode="w")
    print('csv 저장 완료')

def change_datetime(df):
    #df['시점'] = pd.to_datetime(df['시점'],format="%m%Y")
    return df

def create_season(df):
    season = ['봄','여름','가을','겨울']
    conditions = [ (3 <= df['월']) & (df['월'] <= 5), (6 <= df['월']) & (df['월'] <= 8),
                  (9 <= df['월']) & (df['월'] <= 11), (12 <= df['월']) & (df['월'] <= 2)]
    col = ['월']

    df['계절'] = np.select(conditions,season,default=np.nan)
    return df