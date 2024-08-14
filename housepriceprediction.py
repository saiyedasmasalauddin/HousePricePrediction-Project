import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import joblib

#reading data from Data.csv file
ds= pd.read_csv("C:\\Users\\NAMIRA $ ASMA\\OneDrive\\ASMA DATA SCIENCE\\House price prediction project\\HousePricePrediction.csv")
ds= ds.drop(columns=['types'],axis = 1)
df=pd.DataFrame(ds)
print('columns:',len(ds))

# Check for null values in each column using the isnull() method
null_values = df.isnull().any()
print("Null values in each column:", null_values)


#splitting dependent and independent variables
X = df.drop(columns=['selling price(in lacs)'],axis = 1)
y = df['selling price(in lacs)']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=(55))

# #Feature Scaling of datasets
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
ss= StandardScaler() 
##ss=MinMaxScaler() #object of MinMaxScaler is created
x1_train= ss.fit_transform(X_train)#applies the math formula of min max scaler
x1_test= ss.fit_transform(X_test)

# Create and train a ridge regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.1)  # Choose an appropriate alpha value
ridge_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_reg.predict(X_test)
# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
print("model score:",ridge_reg.score(X_test, y_test))
rmse = mean_squared_error(y_test, y_pred)**0.5
print("RMSE:", rmse)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df)

#Displaying the accuracy score
print("Train Score: ", ridge_reg.score(X_train, y_train))
print("Test Score: ",ridge_reg.score(X_test, y_test))

new_house = pd.DataFrame({'squarefeet area':[2000],'BHK':[2],'year built':[2023]})
predicted_price = ridge_reg.predict(new_house)
print("Predicted Price:",predicted_price[0])


joblib.dump(ridge_reg,"asma.model_ml")



import pickle
pickle.dump(ridge_reg,open('housepriceprediction.pkl','wb'))

model= pickle.load(open('housepriceprediction.pkl','rb'))
print(ridge_reg.predict([[2000,2,2023]]))




