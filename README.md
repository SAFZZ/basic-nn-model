## EXP.NO : 1 
## DATE :

# Developing a Neural Network Regression Model

## AIM:

To develop a neural network regression model for the given dataset.

## THEORY:

Neural Networks
Neural networks are a set of algorithms, modeled loosely after the human brain, that are designed to recognize patterns. They interpret sensory data through a kind of machine perception, labeling or clustering raw input. The patterns they recognize are numerical, contained in vectors, into which all real-world data, be it images, sound, text or time series, must be translated.

Regression model
A regression model provides a function that describes the relationship between one or more independent variables and a response, dependent, or target variable. For example, the relationship between height and weight may be described by a linear regression mode.

## Neural Network Model:

![image](https://user-images.githubusercontent.com/75413726/187075533-8dc7904c-1988-44a3-887f-c985661bd713.png)

## DESIGN STEPS:

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM:
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
worksheet = gc.open('StudentsData').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.dtypes
X=df[['Input']].values
X
Y=df[['Output']].values
Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
X_train
scaler=MinMaxScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train1=scaler.transform(X_train)
X_test1=scaler.transform(X_test)
X_train1
ai_brain=Sequential([
    Dense(4,activation='relu'),
    Dense(6,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,Y_train,epochs=8000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
ai_brain.evaluate(X_test1,Y_test)
input=[[100]]
input1=scaler.transform(input)
input1.shape
ai_brain.predict(input1)
```
## Dataset Information:

![image](https://user-images.githubusercontent.com/75413726/187073217-1465db96-3d6d-4c1b-80e4-23ac7e3010be.png)

## OUTPUT:

### Training Loss Vs Iteration Plot:

![image](https://user-images.githubusercontent.com/75413726/187073791-a50eecaf-225d-449d-b816-738ff1a7f45d.png)

### Test Data Root Mean Squared Error:

0.00301834917627275

### New Sample Data Prediction:

![image](https://user-images.githubusercontent.com/75413726/187073828-30bb5c9c-3a38-4d83-8806-113fdd4b738b.png)

## RESULT:

Succesfully created and trained a neural network regression model for the given dataset.
