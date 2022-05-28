import pandas as pd
def load_csv(csv_name):
    data = pd.read_csv(f'data/{csv_name}.csv', encoding='euc-kr')
    return data

def join_data(df1,df2,dataName):
    data = pd.concat([df1,df2[f'{dataName}']],axis=1)
    return data
