# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 15:11:39 2018

@author: jpdub
"""

from pandas import read_csv
from datetime import datetime
# load data
#def parse(x):
#	return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('C:\\Users\jpdub\injprod1.csv', index_col=0)
#dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['inj_vol_tot', 'days_inj', 'inj_rate_sum', 'inj_avg_pres_sum', 'inj_max_pres_sum', 'oil_prod', 'gas_prod', 'water_prod', 'water_cut']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['water_prod'].fillna(0, inplace=True)
# drop the first 24 hours
#dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('injprod_edit.csv')