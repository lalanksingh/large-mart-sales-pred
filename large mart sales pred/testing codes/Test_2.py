# Random Forest

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# convert list to list of lists
def extractDigits(lst):
    return [[el] for el in lst]

# Importing the dataset
dataset = pd.read_csv('Train.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# importing test.csv dataset
test_dataset = pd.read_csv('Test.csv')
xt = test_dataset.iloc[:, 1:].values
ii = test_dataset.iloc[:, 0].values
oi = test_dataset.iloc[:, 6].values
# print(ii,'\n',oi)

# replacing LF to low fat in column
n=0
for n in range(len(X)) :
    if X[n][1]=='LF' :
        X[n][1]='Low Fat'
    if X[n][1] == 'reg':
        X[n][1] = 'Regular'
    n=n+1

n=0
for n in range(len(xt)) :
    if xt[n][1]=='LF' :
        xt[n][1]='Low Fat'
    if xt[n][1] == 'reg':
        xt[n][1] = 'Regular'
    n=n+1


# Taking care of missing data
from sklearn.impute import SimpleImputer
a= [0] #row index to be applied
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, a])
X[:, a] = imputer.transform(X[:, a])
imputer.fit(xt[:, a])
xt[:, a] = imputer.transform(xt[:, a])

a= [7] #row index to be applied
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Medium')
imputer.fit(X[:, a])
X[:, a] = imputer.transform(X[:, a])
imputer.fit(xt[:, a])
xt[:, a] = imputer.transform(xt[:, a])


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
xt = np.array(ct.fit_transform(xt))
# print(X[0],'\n',len(X[0]))
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [22,23,24,25,26])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
xt = np.array(ct.fit_transform(xt))
# print(xt[1])

# Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X_train = sc_X.fit_transform(X)
# y_train = sc_y.fit_transform(y)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred= regressor.predict(xt)

# # metrics evluation
# from sklearn.metrics import r2_score, mean_absolute_error
# a = r2_score(y_test, y_pred)
# b = mean_absolute_error(y_test,y_pred)
# print('R square score = ',a)
# print('mean absolute error = ',b)



# Driver code
n=0
d=[]
e=[]
for n in range (len(ii)):
    d.append(ii[n])
    d.append(oi[n])
    d.append(y_pred[n])
    e.append(d)
    d=[]
    n=n+1

# print(e)

# ii = extractDigits(ii)
# oi = extractDigits(oi)
# print(ii)
file = open("sub_of_2.csv", "w+", newline ='')
with file:
    write = csv.writer(file)
    write.writerow(['Item_Identifier','Outlet_Identifier', 'Item_Outlet_Sales' ])
    write.writerows(e)
