import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import numpy as np
import pickle


file=pd.read_csv("dengue.csv")
file=file.drop(['dengue.p_i_d','dengue.date_of_fever','dengue.residence'], axis=1)
file1=file
## Numeric columns
imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
list1 = ['dengue.wbc','dengue.hemoglobin','dengue._hematocri' ,'dengue.platelet']
for col in list1:
   file1[col] = imputer.fit_transform(file1[col].values.reshape(-1,1))

## string or bool Columns

imputer1= SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
list2 = ['dengue.days','dengue.servere_headche','dengue.joint_muscle_aches','dengue.dengue'] 
for col in list2:
   file1[col] = imputer1.fit_transform(file1[col].values.reshape(-1,1))


if file1['dengue.days'].str.contains('days').any():
  file1['dengue.days'] = file1['dengue.days'].str.replace('days','')
if file1['dengue.days'].str.contains('2 weeks').any():
  file1['dengue.days'] = file1['dengue.days'].str.replace('2 weeks','14')
if file1['dengue.days'].str.contains('12 months').any():
  file1['dengue.days'] = file1['dengue.days'].str.replace('12 months','365')
if file1['dengue.days'].str.contains('3--4').any():
  file1['dengue.days'] = file1['dengue.days'].str.replace('3--4','4')
file1['dengue.days']= file1['dengue.days'].astype(int)

# labelEncoder
ln = LabelEncoder() 
list = ['dengue.servere_headche','dengue.pain_behind_the_eyes','dengue.joint_muscle_aches','dengue.metallic_taste_in_the_mouth','dengue.appetite_loss','dengue.addominal_pain','dengue.nausea_vomiting','dengue.diarrhoea','dengue.dengue']
for col in list:
  file1[col] = ln.fit_transform(file1[col]) 

#  scaling
scaler = MinMaxScaler()
file1[['dengue.days','dengue.current_temp','dengue.wbc','dengue.hemoglobin','dengue._hematocri','dengue.platelet']] = scaler.fit_transform(file1[['dengue.days','dengue.current_temp','dengue.wbc','dengue.hemoglobin','dengue._hematocri','dengue.platelet']] )


x = file1.iloc[:, :-1]
y = file1.iloc[:, -1]

# # smote
sm = SMOTE(sampling_strategy='minority')
x_smote, y_smote =sm.fit_resample(x,y)

# After Smote split data set
xtrain_smote, xtest_smote, ytrain_smote, ytest_smote = train_test_split(x_smote, y_smote, test_size= 0.3336, random_state=0,stratify=y_smote)


# Instantiate the model
DT_model =DecisionTreeClassifier()
DT = { 'criterion':['gini'],'max_depth': [5]}
grid_DT= GridSearchCV(estimator = DT_model, param_grid = DT, cv = 5, n_jobs=-1)

# Instantiate the model
grid_DT.fit(xtrain_smote,ytrain_smote)

# # Make pickle file of our model
pickle.dump(grid_DT, open("model.pkl", "wb"))



