import pandas as pd
def load_csv(csv_name):
    data = pd.read_csv(f'data/{csv_name}.csv', encoding='euc-kr')
    return data

def SBHI_to_csv(data):
  data.set_index('시점',inplace=True)
  data = data.sort_values(by='시점',ascending=True)
  return data
