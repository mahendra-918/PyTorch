import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')

df.drop(columns=['id','Unnamed: 32'],inplace=True)



X_trian,X_test,y_train,y_test = train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.2)

scaler = StandardScaler()
X_trian = scaler.fit_transform(X_trian)
X_test = scaler.transform(X_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

X_trian_tensor = torch.from_numpy(X_trian)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)


class mySimpleNN():
    def __init__(self,X):
        self.weights = torch.rand(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    def forward(self,X):
        z = torch.matmul(X,self.weights)+self.bias
        y_pred = torch.sigmoid(z)
        return y_pred
    
    def loos_fucntion(self,y_pred,y):
        eplison = 1e-7
        y_pred = torch.clamp(y_pred,eplison,1-eplison)
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss


learning_rate = 0.1
epochs = 25

model = mySimpleNN(X_trian_tensor)

for epoch in range(epochs):
    y_pred = model.forward(X_trian_tensor)
    loss = model.loos_fucntion(y_pred,y_train_tensor)
    loss.backward()
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad
    
    model.weights.grad.zero_()
    model.bias.grad.zero_()
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')


