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

import preprocessing as pp

employ_data = pp.load_csv('고용수준실적SBHI')
business_data = pp.load_csv('경기전반실적SBHI')
domestic_data = pp.load_csv('내수판매실적SBHI')
export_data = pp.load_csv('수출실적SBHI')
print(export_data)

general_data = pp.join_data(business_data,employ_data,"고용수준실적SBHI")
general_data = pp.join_data(general_data,domestic_data,"내수판매SBHI")
general_data = pp.join_data(general_data,export_data,"수출실적SBHI")
print(general_data)

# general_data.to_csv("data/최종데이터.csv")

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
