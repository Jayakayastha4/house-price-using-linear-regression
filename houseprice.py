
import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
df=pd.read_csv(r"C:\Users\HP\Downloads\1553768847-housing.csv")
#print(df.head())
print(df.shape)
#cr_matrix=df.corr()
#print(cr_matrix)
#print(df.isnull().sum())
most_frequent = df['total_bedrooms'].mode()[0]
df.fillna({'total_bedrooms':most_frequent}, inplace=True)
print(df.isnull().sum())
df = pd.get_dummies(df, columns=['ocean_proximity'])
#print(df.head())
correlation=df.corr()
#print(correlation)
import seaborn as sns
import matplotlib.pyplot as plt
"""
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()
"""
Y=df['median_house_value']
X=df.drop('median_house_value',axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
error=mean_squared_error(Y_test,y_pred)
print(error)
### now use features which are highly correlated 
df['room_per_household']=df['total_rooms']/df['households']
df['Bedrooms_per_Room']=df['total_bedrooms']/df['total_rooms']
#print(df.head())
X_new=df.drop(['housing_median_age','total_rooms','total_bedrooms','population','households','median_house_value'],axis=1)
x_tr,x_tt,Y_train,Y_test=train_test_split(X_new,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_tr,Y_train)
y_pred1=model.predict(x_tt)
error1=mean_squared_error(Y_test,y_pred1)
print(error1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
## using standered scaler
X_train_scaled = scaler.fit_transform(x_tr)
X_test_scaled = scaler.transform(x_tt)
model.fit(X_train_scaled,Y_train)
y_pred2=model.predict(X_test_scaled)
error3=mean_squared_error(Y_test,y_pred2)
print(error3)
#using polynomial 
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train_scaled)
model.fit(X_poly, Y_train)
X_polytest = poly.fit_transform(X_test_scaled)
ypred3=model.predict(X_polytest)
error4=mean_squared_error(Y_test,ypred3)
print(error4)