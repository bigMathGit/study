import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# csv 파일 읽기
df = pd.read_csv("creditcard.csv")

# Amount_max_fraud 컬럼 생성
df['Amount_max_fraud'] = 1
df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0

#Drop all of the features that have very similar distributions between the two types of transactions.
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)

#Based on the plots above, these features are created to identify values where fraudulent transaction are more common.
# fraudulent transaction 이 더 잘나타는 방향으로 자료 정리 V1_~ V21_ 생성
df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)

# Create a new feature for normal (non-fraudulent) transactions.
# Normal 컬럼 생성 Class가 0 이면 1, Class가 1이면 0
df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0

# Rename 'Class' to 'Fraud'
# Class -> Fraud 로 명 변환
df = df.rename(columns = {'Class' : 'Fraud'})

# 492 fraudulent transactions, 284,315 normal transactions.
# 0.172% of transactions were fraud.
print(df.Normal.value_counts())
print()
print(df.Fraud.value_counts())

# max column 수 설정
pd.set_option('display.max_columns',  101)
# print(df.head())

# Create dataframes of only Fraud and Normal transactions.
Fraud = df[df.Fraud == 1]
Normal = df[df.Normal == 1]
print('Fraud  : ', len(Fraud))
print('Normal : ', len(Normal))

# Set X_train equal to 80% of the fraudulent transactions.
FraudSample  = Fraud.sample(frac=0.8)
NormalSample = Normal.sample(frac=0.8)
count_Frauds = len(FraudSample)
# Add 80% of the normal transactions to X_train.
for_train = pd.concat([FraudSample, NormalSample], axis=0)

# X_test contains all the transaction not in X_train.
for_test = df.loc[~df.index.isin(for_train.index)]

print('len(for_train)  : ',len(for_train))
print('len(for_test)   : ',len(for_test))

#Shuffle the dataframes so that the training is done in a random order.
for_train = for_train.sample(frac=1).reset_index(drop=True)
for_test = for_test.sample(frac=1).reset_index(drop=True)
#for_test = np.random.shuffle(for_test)

# Add our target features to y_train and y_test.
X_train = for_train.drop(['Fraud', 'Normal'], axis = 1)
# Drop target features from X_train and X_test.
#  Fraud, Normal 컬럼 drop
y_train = for_train[['Fraud', 'Normal']]

# Add our target features to y_train and y_test.
X_test = for_test.drop(['Fraud', 'Normal'], axis = 1)
# Drop target features from X_train and X_test.
#  Fraud, Normal 컬럼 drop
y_test = for_test[['Fraud', 'Normal']]

#Check to ensure all of the training/testing dataframes are of the correct length
print('len(X_train) : ',len(X_train))
print('len(y_train) : ',len(y_train))
print('len(X_test)  : ',len(X_test))
print('len(y_test)  : ',len(y_test))

# In [26]
'''
Due to the imbalance in the data, ratio will act as an equal weighting system for our model. 
By dividing the number of transactions by those that are fraudulent, ratio will equal the value that when multiplied
by the number of fraudulent transactions will equal the number of normal transaction. 
Simply put: # of fraud * ratio = # of normal
'''

ratio = len(X_train) / count_Frauds
print('ratio :', ratio)

y_train.Fraud *= ratio
y_test.Fraud *= ratio

#Names of all of the features in X_train.
features = X_train.columns.values
print('features : ',features)

#Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
#this helps with training the neural network.
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    # print('feature :',feature , 'mean : ', mean , 'std :', std)
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std

'''
Train the Neural Net
'''

# In [28]
# Split the testing data into validation and testing sets
split = int(len(y_test)/2)
print('split : ', split)

inputX = X_train.as_matrix()
inputY = y_train.as_matrix()
inputX_valid = X_test.as_matrix()[:split]
inputY_valid = y_test.as_matrix()[:split]
inputX_test = X_test.as_matrix()[split:]
inputY_test = y_test.as_matrix()[split:]

print('type(inputX)  : ', type(inputX))
print('type(X_train) : ', type(X_train))

print('y_train.Normal.value_counts() :',y_train.Normal.value_counts())
print('y_train.Fraud.value_counts() :',y_train.Fraud.value_counts())

print('y_test.Normal.value_counts() :',y_test.Normal.value_counts())
print('y_test.Fraud.value_counts() :',y_test.Fraud.value_counts())

print('C inputY_valid', np.where(inputY_valid[:, 0] > 0, 1, 0).sum())
print('C inputY_test ', np.where(inputY_test[:, 0] > 0, 1, 0).sum())

print('inputX :',inputX.shape)
print('inputY :',inputY.shape)
print('inputX_valid :',inputX_valid.shape)
print('inputY_valid :',inputY_valid.shape)
print('inputX_test :',inputX_test.shape)
print('inputY_test :',inputY_test.shape)

# Number of input nodes.
input_nodes = 37

# Multiplier maintains a fixed ratio of nodes between each layer.
mulitplier = 1.5

# Number of nodes in each hidden layer
hidden_nodes1 = 18
hidden_nodes2 = round(hidden_nodes1 * mulitplier)
hidden_nodes3 = round(hidden_nodes2 * mulitplier)

# In [31]
# Parameters
training_epochs = 50             # should be 2000, it will timeout when uploading
training_dropout = 0.9          # drop out
display_step = 1                # 10
n_samples = y_train.shape[0]
batch_size = 2048
learning_rate = 0.005           # 하이퍼파라미터

print('======================================================')
print('====================== Keras    ======================')
print('======================================================')

# 모델 구성하기
model = Sequential()
model.add(Dense(hidden_nodes1, input_dim=input_nodes, activation='sigmoid'))
model.add(Dense(hidden_nodes2, activation='sigmoid'))
model.add(Dense(hidden_nodes3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

# 모델 학습과정 설정하기
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

# 모델 학습시키기
hist = model.fit(inputX, inputY, epochs=training_epochs, batch_size=batch_size)

# 모델 평가하기
loss_and_metrics = model.evaluate(inputX_valid, inputY_valid, batch_size=batch_size)
print('loss_and_metrics : ' + str(loss_and_metrics))

fig, loss_ax =  plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['acc'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

print('======================================================')
print('====================== Keras    ======================')
print('======================================================')
