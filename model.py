
# importing packages 
import numpy as np 
import pandas as pd 
import pickle 

# loading data set 
data = pd.read_csv('student-mat.csv',delimiter= ";")
selected_features = ['Medu','Fedu','health','absences','G1','G2','G3']
df = data[selected_features]

# train test split 
from sklearn.model_selection import train_test_split
X = df.drop("G3",axis =1)
y = df['G3']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 101)

# model building 
from sklearn.linear_model import LinearRegression 
regression_model = LinearRegression()
regression_model.fit(X_train,y_train)

# using pickle to save the model to disk 
pickle.dump(regression_model,open('reg_model.pkl','wb'))