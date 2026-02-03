# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Input the Dataset
Load house features (size, rooms, location) and target values (house price and number of occupants).

2. Preprocess and Split Data
Split the dataset into training and testing sets and apply feature scaling.

3. Train the Model
Train the SGD Regressor (using MultiOutputRegressor) on the training data.

4. Predict and Evaluate
Predict house price and number of occupants for test/new data and evaluate using Mean Squared Error (MSE).
 

## Program:
```
name: Tejashree M
register number: 252225220115
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: 
RegisterNumber:  
*/
```
```
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


X = np.array([
    [800, 2, 5],
    [1200, 3, 7],
    [1500, 4, 8],
    [600, 1, 4],
    [2000, 5, 9],
    [1000, 3, 6],
    [1800, 4, 8]
])


y = np.array([
    [120000, 3],
    [200000, 4],
    [250000, 5],
    [90000, 2],
    [320000, 6],
    [180000, 4],
    [280000, 5]
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", MultiOutputRegressor(
        SGDRegressor(max_iter=1000, tol=1e-3)
    ))
])

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



new_house = np.array([[1400, 3, 7]])
prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0][0])
print("Predicted Number of Occupants:", round(prediction[0][1]))



## Output:
<img width="464" height="73" alt="Screenshot 2026-02-03 131144" src="https://github.com/user-attachments/assets/d4f1f653-3802-4a3e-bc0f-fb29b71a3a81" />

Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
