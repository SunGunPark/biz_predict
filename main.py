# 기본 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 머신러닝 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datetime import datetime
import preprocessing as pp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager, rc
import seaborn as sns

font_path = "C:/Windows/Fonts/H2GTRM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

employ_data = pp.load_csv('고용수준실적SBHI')
business_data = pp.load_csv('경기전반실적SBHI')
domestic_data = pp.load_csv('내수판매실적SBHI')
export_data = pp.load_csv('수출실적SBHI')
fund_data = pp.load_csv('자금사정실적SBHI')
sales_data = pp.load_csv('영업이익실적SBHI')
esi_data = pp.load_csv('경제심리지수')
all_data = pp.load_csv('데이터셋')
rm_data = pp.load_csv('원자재_수입동향')
#print(esi_data)

general_data = pp.join_data(business_data,employ_data,"고용수준실적SBHI")
general_data = pp.join_data(general_data,domestic_data,"내수판매SBHI")
#general_data = pp.join_data(general_data,export_data,"수출실적SBHI")
general_data = pp.join_data(general_data,fund_data,"자금사정실적SBHI")
general_data = pp.join_data(general_data,sales_data,"영업이익실적SBHI")
general_data = pp.create_nan_data(general_data)
general_data = pp.delete_nan_data(general_data)
general_data = pp.change_datetime(general_data)
#print(general_data.corr())

#print(general_data.info)
# general_data.plot(x='시점')

general_data['시점'] = general_data['시점'].astype(str)
all_data['시점'] = all_data['시점'].astype(str)
rm_data['시점'] = rm_data['시점'].astype(str)
all_data = pp.join_data_2(all_data,rm_data)
all_data.loc[all_data['시점']=='2015.1','시점']='2015.10'
all_data.loc[all_data['시점']=='2016.1','시점']='2016.10'
all_data.loc[all_data['시점']=='2017.1','시점']='2017.10'
all_data.loc[all_data['시점']=='2018.1','시점']='2018.10'
all_data.loc[all_data['시점']=='2019.1','시점']='2019.10'
all_data.loc[all_data['시점']=='2020.1','시점']='2020.10'
all_data.loc[all_data['시점']=='2021.1','시점']='2021.10'
all_data['시점'] = pd.to_datetime(all_data['시점'])
all_data['년'] = all_data['시점'].dt.year
all_data['월'] = all_data['시점'].dt.month
all_data['계절'] = np.select([(all_data['월'] <= 2),(all_data['월'] <= 5),
                            (all_data['월'] <= 8),(all_data['월'] <= 11),
                            (all_data['월'] == 12)],
                           ['겨울','봄','여름','가을','겨울'], default=np.nan)

print(all_data)

pp.save_df_to_csv(all_data,"최종데이터")


#plt.plot(general_data['시점'],general_data['고용수준실적SBHI'])
plt.scatter(all_data['시점'],all_data['고용수준실적SBHI'])
plt.title('고용수준실적SBHI 산점도 그래프')
#plt.show()

x_train = []
x_valid = []
y_train = []
y_valid = []

def ml_fit(model):
  model.fit(x_train, y_train)
  pred = model.predict(x_valid)
  accuracy = accuracy_score(pred, y_valid)
  print(f'정확도 : {accuracy*100:.3f}%')
  return model

#model = ml_fit(RandomForestClassifier(
#    n_estimators=50, criterion="entropy",
#     max_depth=5, oob_score=True, random_state=10))

#model = ml_fit(LogisticRegression(solver='lbfgs'))

#model = ml_fit(SVC(gamma='scale'))

#model = ml_fit(KNeighborsClassifier())

#model = ml_fit(GaussianNB())

#model = ml_fit(DecisionTreeClassifier())

#model = ml_fit(GaussianNB())

#model = ml_fit(DecisionTreeClassifier())
