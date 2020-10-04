# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 08:28:10 2020

@author: svenn
"""
#Import the standard libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm


#Add the folder where the python code files are stored
import sys
sys.path.append('C:/Users/svenn/Documents/Werk/PythonProjects/multivariate_fc/multi_fc_code/')

#Import the folders
import ARX_model
import MinT_model

#%%
###############################################################################
#                       ARX EXAMPLES
###############################################################################

#Simulate the three time series with the arma simulator (add constant to make them positive/o.w. log operation fails)
c = 10; c_ex = 20
serie_1 = sm.tsa.arma_generate_sample(ar = [1, 0.75, 0.3, -0.2], ma = [1], nsample=500) + c
serie_2 = sm.tsa.arma_generate_sample(ar = [1, 0.2, 0.15], ma = [1], nsample=500) + c
serie_3 = sm.tsa.arma_generate_sample(ar = [1, 0.4], ma = [1], nsample=500) + c

#put them in a dataframe
time_series_df = pd.DataFrame(np.vstack([serie_1,serie_2, serie_3]).T, columns = ['a', 'b', 'c'])

#Simulate some external data
ex = sm.tsa.arma_generate_sample(ar = [1, 0.4,0.2], ma = [1], nsample=500) + c_ex
ex = pd.DataFrame(ex, columns = ['x_1'])

#Set the parameters
prepare = True; log = True; stationarity = False; h = 2

#Initiate the model
ARX = ARX_model.ARX(time_series_df)

#Fit the model (there is also the option to add external variables, via ex=..)
fit = ARX.fit(prepare = prepare, log = log , stationarity = stationarity, lags = {'a': 2, 'b':3, 'c':5}, ex = ex, ex_thresholds = {'x_1':0.6})

#forecast with the fitted model
ARX_forecast = fit.forecast(h=1, ex_test=pd.DataFrame([[20]], columns = ['x_1']))




#%%
###############################################################################
#                       MinT EXAMPLES
###############################################################################


#Hierarchical class example --> hierarchical structure starts with bottom level
hierarchical_levels = [['Product_type'], ['Product']]
bottom_level = ['Product']
hierarchical_structure = pd.DataFrame([['Cola','Water', 'Egg', 'bread'], ['Drinks', 'Drinks', 'Food','Food']], index = ['Product', 'Product_type'])

#Sort the structure such that the alphabetic order of the first row is leading
hierarchical_structure = hierarchical_structure.sort_values(list(hierarchical_structure.index), axis=1)

#Simulate the data for the 4 bottom series
serie_1 = sm.tsa.arma_generate_sample(ar = [1, 0.75, 0.3, -0.2], ma = [1], nsample=500) + 10
serie_2 = sm.tsa.arma_generate_sample(ar = [1, 0.2, 0.15], ma = [1], nsample=500) + 10
serie_3 = sm.tsa.arma_generate_sample(ar = [1, 0.4], ma = [1], nsample=500) + 10
serie_4 = sm.tsa.arma_generate_sample(ar = [1, -0.4,0.2], ma = [1], nsample=500) + 10

#Create the actual data at the lowest level
bottom_actuals = pd.DataFrame(np.vstack([serie_1, serie_2, serie_3, serie_4]).T, columns = pd.MultiIndex.from_frame(hierarchical_structure.T))

#Now add all other levels of aggregation to the data
actuals = []
for level in hierarchical_levels: #We already have the lowest level, hence, do not 
    actuals.append(bottom_actuals.groupby(level = level, axis = 1).sum())
actuals = pd.concat(actuals, axis =1)

#Also add the total actuals
actuals.insert(0, column = 'Total', value = bottom_actuals.sum(axis=1))

#Initiate the MinT object     
MinT_test = MinT_model.MinT(hierarchical_structure, bottom_level, hierarchical_levels)        

#Generate the forecasts for the bottom level of the hierarchical structure
MinT_forecasts = MinT_test.fit_forecast_rpy2(actuals,1)

##############################################################################
###############################################################################
#%%Grouped class example (multiple hierarchical structures combined)

#Define the parameters (A grouped structure with a bottom level[1,2] and two non hierarchical levels 1 and 2)
hierarchical_levels = [['Product'], ['Debtor_id'], ['Product','Debtor_id']]
bottom_level = ['Product', 'Debtor_id']
hierarchical_structure = pd.DataFrame([['Egg','Egg','Water', 'Water'], ['1', '2', '1', '2']], index = ['Product','Debtor_id'])

#Sort the structure such that the alphabetic order of the first row is leading
hierarchical_structure = hierarchical_structure.sort_values(list(hierarchical_structure.index), axis=1)

#Simulate 4 arima time series (the four bottom time series)
serie_1 = sm.tsa.arma_generate_sample(ar = [1, 0.75, 0.3, -0.2], ma = [1], nsample=500) + 10
serie_2 = sm.tsa.arma_generate_sample(ar = [1, 0.2, 0.15], ma = [1], nsample=500) + 10
serie_3 = sm.tsa.arma_generate_sample(ar = [1, 0.4], ma = [1], nsample=500) + 10
serie_4 = sm.tsa.arma_generate_sample(ar = [1, -0.4,0.2], ma = [1], nsample=500) + 10

#Create the actual data at the lowest level (bottom_level)
bottom_actuals = pd.DataFrame(np.vstack([serie_1, serie_2, serie_3, serie_4]).T, columns = pd.MultiIndex.from_frame(hierarchical_structure.T))

#Now create the actuals at all desired hierarchical levels
actuals = []
for level in hierarchical_levels:
    actuals.append(bottom_actuals.groupby(level = level, axis=1).sum())
actuals = pd.concat(actuals, axis=1)

#Also add the total actuals
actuals.insert(0, column = 'Total', value = bottom_actuals.sum(axis=1))

#Initiate the MinT object     
MinT_test = MinT_model.MinT(hierarchical_structure, bottom_level, hierarchical_levels)   

#Generate the reconcilled forecasts (with MinT) for the bottom level of the hierarchical structure
#The base forecasts are now created within the function with the ARX model (no external vars and default preprocess)
MinT_fc = MinT_test.fit_forecast_rpy2(actuals, h = 2)

#%%
#Also possible to do it without the help of the hts package, so only with python code

#Fit the model
MinT_test.fit_py(actuals)

#Forecast the model
MinT_test.forecast_py(h=10)

#%%It is also possible to independently generate base forecasts and give them as input
#Important the the base forecasts have same column names as actuals
#Initiate the ARX model
ARX = ARX_model.ARX(actuals)

#Fit the ARX model (there is also the option to add external variables, via ex=..)
fit_model = ARX.fit(prepare = True, log = True , stationarity = True)

#Generate the base forecasts for all columns with the fitted ARX model
base_forecasts = fit_model.forecast(h=2).forecasts

#However when you want to use you own base forecasts, you should alos input the corresponding fittedvalues
MinT_fc_own_base_fc = MinT_test.fit_forecast_rpy2(actuals, h = 2, base_forecasts = base_forecasts, fittedvalues = fit_model.fittedvalues)