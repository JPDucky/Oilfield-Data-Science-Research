# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 02:30:53 2018

@author: jpdub
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('GTKAgg')
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]        # create var n_vars, set number of variables to that in array
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
##Create a differenced dataset        inverse  =============== reverse the function
#def difference(dataset, interval=1):
#    diff = list()
#    for i in range(interval, len(dataset)):
#        value = dataset[i] - dataset[i - interval]
#        diff.append(value)
#    return dataset1(diff)
#
##Invert differenced value
#def inverse_difference(history, yhat, interval=1):
#    return yhat + history[-interval]


# load dataset
dataframe = read_csv('C:\\Users\\jpdub\\Documents\\School\\PETE 4998\\IMMU.csv', header=0, index_col=0)
#print(dataframe.head(5))
#dataset1 = dataframe[['cum_oil_prod', 'cum_gas_prod', 'cum_water_prod', 'cum_water_inj']]
dataset1 = dataframe#[['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']] #import all columns for stock
var = dataframe[['High']]
#print(dataset1.shape) #(376, 4)
values = dataset1.values #[:374]
#print(values[:5])
#print(values.shape)
#dataset.tail(30)

# integer encode direction
#encoder = LabelEncoder()
#values[:,3] = encoder.fit_transform(values[:,3])
#print(values[:5])
#print(values.shape) #(376, 4)

# ensure all data is float
values = values.astype('float32')
print(values.shape)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#print(scaled.shape) #(376, 4)
length = np.size(dataset1, axis=1)
#length = length - 1
#Specify number of lags
n_lag= 50
n_features = length
# frame as supervised learning
reframed = series_to_supervised(scaled, n_lag, 1)
#print(reframed.shape) #(375, 8)
#print(reframed.columns)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[6, 8, 9, 10, 11]], axis=1, inplace=True)
#print(reframed.shape) #(375, 5)
#print(reframed) #yup all 5 are there, dunno why reframed shape is 8{[?]} == its the (t-1) variables lol

# split into train and test sets
values1 = reframed.values
n_train = int(values1.shape[0]*0.6)
train = values1[:n_train, :]
test = values1[n_train:, :]

# split into input and outputs
n_obs = n_lag * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]

# reshape input to be 3D [samples, timesteps, features] <<<<<<<================ yooooooooooooooooooooooooo read this shit!!!!!!
train_X = train_X.reshape((train_X.shape[0], n_lag, n_features))
test_X = test_X.reshape((test_X.shape[0], n_lag, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# hyperparameters
nodes = 29
#print(train_X.shape[])
#print(train_X.shape) #(250,3)   [0]= 250 , [1]= 3
#input_shape = (train_X.shape[1], train_X.shape[2])
#print(input_shape) #(1, 6)

 # design network
model = Sequential()   #                (   1      ,     6   )
model.add(LSTM(nodes, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])

# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size=64, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#print(len(history.history))
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make a prediction
testPredict = model.predict(test_X) #generates output predictions for the input samples
trainPredict = model.predict(train_X)
#print('test predict')
#print(testPredict.shape) #(125, 1)
#print(test_X.shape)
test_X = test_X.reshape((test_X.shape[0], n_lag*n_features)) 
train_X = train_X.reshape((train_X.shape[0], n_lag*n_features))
#print('testx')
#print(test_X.shape) #(125, 5)

# invert scaling for forecast
inv_testPredict = concatenate((testPredict, test_X[:, -5:]), axis=1) #knocks off the first column and sticks the two variables together in the same array
inv_trainPredict = concatenate((trainPredict, train_X[:, -5:]), axis=1)
#print('tpd')
#print(inv_testPredict.shape)
#print(inv_testPredict.shape) #(125, 5)
inv_testPredict = scaler.inverse_transform(inv_testPredict)
inv_testPredict = inv_testPredict[:,0]
inv_trainPredict = scaler.inverse_transform(inv_trainPredict)
inv_trainPredict = inv_trainPredict[:,0]
#print(inv_testPredict)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_ytest = concatenate((test_y, test_X[:, -5:]), axis=1)
inv_ytest = scaler.inverse_transform(inv_ytest)
inv_ytest = inv_ytest[:,0]
print(train_y.shape)
train_y = train_y.reshape((len(train_y), 1))
inv_ytrain = concatenate((train_y, train_X[:, -5:]), axis=1)
inv_ytrain = scaler.inverse_transform(inv_ytrain)
inv_ytrain = inv_ytrain[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_ytest, inv_testPredict))
print('Test RMSE: %.3f' % rmse)
print(inv_ytest[-1])
#Shift the plot for n_lag
pad = np.arange(n_lag)
for i in pad[0:n_lag]:
    pad[i] = 0
#print(pad)
#print(pad.shape)
#pad = pad.reshape(pad(n_lag), (len(n_lag)))
#put the predicted train and test datasets together
predicted_vals = concatenate((pad, inv_ytrain, inv_ytest), axis=0)

#plt.figure(figsize=(20,10))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(predicted_vals, 'r')
#print(predicted_vals.shape)
#print(var.shape)
plt.plot(var, 'g')#need to get in inverse--> done
#print(max(predicted_vals))
#print(min(predicted_vals))
print('Predicted Max: %.3f' % predicted_vals[-1])

