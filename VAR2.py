# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:36:16 2018

@author: jpdub
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

## convert series to supervised learning
#def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#	n_vars = 1 if type(data) is list else data.shape[1]
#	df = DataFrame(data)
#	cols, names = list(), list()
#	# input sequence (t-n, ... t-1)
#	for i in range(n_in, 0, -1):
#		cols.append(df.shift(i))
#		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
#	# forecast sequence (t, t+1, ... t+n)
#	for i in range(0, n_out):
#		cols.append(df.shift(-i))
#		if i == 0:
#			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
#		else:
#			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
#	# put it all together
#	agg = concat(cols, axis=1)
#	agg.columns = names
#	# drop rows with NaN values
#	if dropnan:
#		agg.dropna(inplace=True)
#	return agg

df = pd.read_csv("red_river_b_datacum.csv", header=0, index_col=0) #parse_dates=[['Date']])
df_new = df[['cum_oil_prod', 'cum_gas_prod', 'cum_water_prod', 'cum_water_inj', 'sum_max_pres']]#, 'gas_prod', 'water_prod', 'water_cut', 'sum_tot_inj', 'days_inj', 'sum_inj_rate', 'thick', 'phi', 'k', 'compr', 'Swi', 'oil_dens', 'pres_aband', 'viscos_bp', 'fvf', 'press_init']]
print(df_new.head())
df_new = df_new[:350]
data = df_new
#dataset.columns = ['sum_tot_inj', 'days_inj', 'sum_inj_rate', 'sum_inj_pres', 'sum_max_pres', 'oil_prod', 'gas_prod', 'water_prod', 'water_cut', 'cum_water_inj', 'cum_oil_prod', 'cum_gas_prod', 'cum_water_prod', 'thick', 'phi', 'k', 'compr', 'Swi', 'oil_dens', 'pres_aband', 'viscos_bp', 'fvf', 'press_init']

#values = data.values
## integer encode direction
#encoder = LabelEncoder()
#values[:,3] = encoder.fit_transform(values[:,3])
## ensure all data is float
#values = values.astype('float32')
## normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
## frame as supervised learning
#reframed = series_to_supervised(scaled, 1, 1)
#

#mdata = sm.datasets.macrodata.load_pandas().data
model = VAR(data)
results = model.fit()
print(results.summary())
#results.plot()

model.select_order(20)
results = model.fit(maxlags=20, ic='aic')

lag_order = results.k_ar
results.forecast(data.values[-lag_order:],6)
results.plot_forecast(10)
#
#irf = results.irf(10)
#irf.plot(orth=False)
#
#irf.plot(impulse='cum_oil_prod')