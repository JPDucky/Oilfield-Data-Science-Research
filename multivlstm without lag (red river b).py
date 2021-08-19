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
#def inverse_difference(history, testPredict, interval=1):
#    return testPredict + history[-interval]


# load dataset
dataframe = read_csv('red_river_b_datacum.csv', header=0, index_col=0)
#print(dataframe.head(5))
dataset1 = dataframe[['cum_oil_prod', 'cum_gas_prod', 'cum_water_prod', 'cum_water_inj']]
#dataset1 = dataframe#[['cum_oil_prod', 'cum_gas_prod', 'cum_water_prod', 'cum_water_inj']] #import all columns for stock

#print(dataset.shape) #(376, 4)
values1 = dataset1.values[:374]
var = values1[:, 0]
  #print(values[:5])
#print(values.shape)
#dataset.tail(30)

# integer encode direction
#encoder = LabelEncoder()
#values[:,3] = encoder.fit_transform(values[:,3])
#print(values[:5])
#print(values.shape) #(376, 4)

# ensure all data is float
values = values1.astype('float32')


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#print(scaled.shape) #(376, 4)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#print(reframed.shape) #(375, 8)
print(reframed.columns)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[5, 6, 7]], axis=1, inplace=True)
#print(reframed.shape) #(375, 5)
print(reframed) #yup all 5 are there, dunno why reframed shape is 8{[?]} == its the (t-1) variables lol



# split into train and test sets
values = reframed.values
#print(values.shape) #(375, 5)
#print(values[-6:])
#print(values[:5])
#print(values.columns)
n_train_hours = int(len(values)*0.6)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
#print(test[:5,:])

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
#print(train_X[:5])
#print(train_y[:5])
#print(train_X.shape) #(250, 4)
#print(train_y.shape) #(250, )
test_X, test_y = test[:, :-1], test[:, -1]
#print(test_X[:5])
#print(test_y[:5])
#print(test_X.shape) # (125, 5)
#print(test_y.shape) # (125, )


# reshape input to be 3D [samples, timesteps, features] <<<<<<<================ yooooooooooooooooooooooooo read this shit!!!!!!
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#print(train_X.shape) # (250 , 1, 5)
#print(train_X[:5])
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#print(test_X.shape) #(125, 1, 5)
#print(test_X[:5])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# hyperparameters
nodes = 49
#print(train_X.shape[])
#print(train_X.shape) #(250,3)   [0]= 250 , [1]= 3
#input_shape = (train_X.shape[1], train_X.shape[2])
#print(input_shape) #(1, 5)

 # design network
model = Sequential()   #                (   1      ,     5   )
model.add(LSTM(nodes, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='nadam')

# fit network
history = model.fit(train_X, train_y, epochs=1, batch_size=12, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#print(len(history.history))
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
#print(history.history.shape)

##Reshape data back to previous shape         these are going to be the inputs
#    #make predictions
#testPredict = model.predict(train_X)
#testPredict = model.predict(test_X)
#print(testPredict) #needs to be (248,4), not (248,1), why second number only 1?
#    #reshape predictions
#testPredict = 
#    #invert predictions
#testPredict = scaler.inverse_transform(train_y)
#train_y = scaler.inverse_transform([testPredict])
#testPredict = scaler.inverse_transform(testPredict)
#test_y = scaler.inverse_transform([test_y])
##print(testPredict)

##sns.set()
#plt.grid()
#plt.xlabel("Time")
#plt.ylabel('Cumulative Oil Production')
##pyplot.plot(scaler.inverse_transform(inv_y))
#plt.plot(testPredict)
#plt.plot(testPredict, 'g')
#plt.show()
print(test_X.shape)
# make a prediction
testPredict = model.predict(test_X) #generates output predictions for the input samples
trainPredict = model.predict(train_X)
#print(testPredict.shape) #(125, 1)
#print(test_X.shape)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2])) 
train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
#print(test_X.shape) #(125, 5)

# invert scaling for forecast
inv_testPredict = concatenate((testPredict, test_X[:, 1:]), axis=1) #knocks off the first column and sticks the two variables together in the same array
inv_trainPredict = concatenate((trainPredict, train_X[:, 1:]), axis=1)
print(inv_testPredict.shape) #(125, 5)
inv_testPredict = scaler.inverse_transform(inv_testPredict)
inv_testPredict = inv_testPredict[:,0]
inv_trainPredict = scaler.inverse_transform(inv_trainPredict)
inv_trainPredict = inv_trainPredict[:,0]
#print(inv_testPredict)

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_ytest = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_ytest = scaler.inverse_transform(inv_ytest)
inv_ytest = inv_ytest[:,0]

train_y = train_y.reshape((len(train_y), 1))
inv_ytrain = concatenate((train_y, train_X[:, 1:]), axis=1)
inv_ytrain = scaler.inverse_transform(inv_ytrain)
inv_ytrain = inv_ytrain[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_ytest, inv_testPredict))
print('Test RMSE: %.3f' % rmse)

#put the predicted train and test datasets together
predicted_vals = concatenate((inv_ytrain, inv_ytest), axis=0)

plt.plot(predicted_vals, 'r')
plt.plot(var, 'g')#need to get in inverse--> done





# inverse scaling for a forecasted value
#def invert_scale(scaler, X, value):
#	new_row = [x for x in X] + [value]
#	array = np.array(new_row)
#	array = array.reshape(1, len(array))
#	inverted = scaler.inverse_transform(array)
#	return inverted[0, -1]








## make predictions
#tPredict = model.predict(inv_y)
#
#tPredictPlot = np.empty_like(reframed)
#tPredictPlot[:, :] = np.nan
#
