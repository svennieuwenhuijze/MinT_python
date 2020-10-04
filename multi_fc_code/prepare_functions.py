# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:41:17 2020

@author: svenn
"""

#%% Import libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm 


#%% Define the preparation function

def prep_data(data, start, stationair=True, log=True, ex=None, ex_struct=None, ex_control = None):
    '''
    Function which prepares the whole data set (train and test),
    However takes decisions only based on the train data set (do not leak future information)
    
    It is possible to log and or difference the data. While the log operation is
    applied to every time series the difference operation is only applied when
    a time series is defined stationary (based on two statistical tests). Moreover
    it is possible to test for stationarity while controlling for external variables.
    These variables should be present in the 
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    stationair : TYPE, optional
        DESCRIPTION. The default is True.
    log : TYPE, optional
        DESCRIPTION. The default is True.
    ex : pd.DataFrame()
        Dataframe with the external variables. should have the same number of
        observations as the data.
    ex_struct : TYPE
        DESCRIPTION.

    Returns
    -------
    Prepared data --> logged and/or differenced.
    Dataframe containing the preprocess per variable
    '''
    
    #Initialize a dataframe for the prepared data
    prepared_data = data.copy()
    
    #Initiliaze a transformation path for every time series, such that it is easy to transform them back to the original state
    preprocess_info = pd.DataFrame(np.zeros((prepared_data.shape[1], 2)), index = prepared_data.columns, columns = ['log', 'differenced'])
            
    #If desired take the log from the data
    if log:
        prepared_data = np.log(data)
        #Save the log transformation in the preprocess path
        preprocess_info.loc[:, 'log'] = True
    else:
        preprocess_info.loc[:,'log'] = False
            
    #Save the logged data
    log_data = prepared_data.copy()
    
    
    #Make every time series level stationair (only use differencing once)
    if stationair:
        #Check which time series is non stationair based on statistical tests for unit root and for stationarity
        #Define the critical p_values
        alpha_adf = 0.05; alpha_kpss = 0.05
        
        #If there are external variables involved control them
        if ex_control is not None:
            #Define a ex_train set --> do not want to leak future information
            ex_train = ex.iloc[:start,:]
            
            #Test for every serie independently
            for col in prepared_data.columns:
                #only use the information from the train data set
                serie = prep_data[col].iloc[:start]
                
                #Remove the effect of external variables for whom we should control the data. First check if the time series is influenced by one of those variables
                serie_control_ex_vars = list(set(ex_struct[col]).intersection(set(ex_control)))
                
                #If it is influenced by one of the control external variables, subtract its effect
                if len(serie_control_ex_vars) > 0:
                    #perform a simple linear regression involving a constant, one lag and the external variable to remove the effect of it
                    X = np.hstack([serie.shift(1).iloc[1:].values.reshape(-1,1), ex_train[serie_control_ex_vars].values[1:]])
                    
                    #Add the constant
                    X = sm.add_constant(X)
                    
                    #Fit the model
                    fit = sm.OLS(serie.iloc[1:], X).fit()
                    
                    #Subtract the effect of the external variables from the serie and obtain a new serie                      
                    serie = serie.iloc[1:]  -  ex[serie_control_ex_vars].values[1:].reshape(-1,1).dot(fit.params[2:])
                
                #Perform both the adf_test and kpss_test on the time series
                adf_test = adfuller(serie); kpss_test = kpss(serie)
                
                #If one of the test sets defines the test as non stationair take appropriate measures
                if (adf_test[1] > alpha_adf) | (kpss_test[1] < alpha_kpss):
                    prepared_data.loc[:, col] = prepared_data.loc[:,col].diff(1)
                    preprocess_info.loc[col,'differenced'] = True
                else:          
                    preprocess_info.loc[col,'differenced'] = False

    #Return the prepared data, logged data, and information about the preprocess (in a dictionary)
    return {'Prepared data': prepared_data, 'Logged data': log_data, 'Preprocess info':preprocess_info}


#%% ARX lags

#Calculate the number of lags to be included in a AR model
def AR_struct(data, start, maxlag = 12, ic = 'aic'):    
    '''
    Function which determines the optimal number of lags to include in 
    the model for every time series

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe which includes the (prepared) time series.
    start : int
        defines the number of periods in the train set --> do not use future
        information.
    maxlag : int
        The maximum number of lags allowed in the model. For NPL time series many lags
        might result in infinite forecasts, because the model has now idea which
        magnitude to give the coefficients of those lags. Moreover, it does not seem
        logical that the values of more than x periods ago influences the time series
        of the next period.
        
        The default is 8.

    Returns
    -------
    AR_struct_dict : dictionary
        DESCRIPTION.

    '''
    #Import libraries
    from statsmodels.tsa.api import AR
    
    #Initiate a dictionary to save all results
    AR_struct_dict = {}
    
    #Calculate the number of lags for every time series
    for col in data.columns:
        try:
            AR_struct_dict[col] = AR(data[col].iloc[:start]).fit(maxlag = maxlag, ic = ic).k_ar
        except:
            AR_struct_dict[col] = 1
            
    #return the dictionary
    return AR_struct_dict


# %% Determine the structure of the external variables

def ex_corr_struct(data, start, ex, ex_thresholds, ex_correlation_way_dict = {}):
    '''
    Function which determines for every time series by which external variables
    its is influenced. Note that the correlation between external variables and 
    time series might be influenced by trend and or seasonality. Hence, it might be
    useful to give some sort of preprocessed data set as input. However, the external
    variables should follow the same preprocess path as the time series.

    Parameters
    ----------
    data : pandas.DataFrame
        DESCRIPTION.
    start : integer
        DESCRIPTION.
    ex : pandas.DataFrame
        DESCRIPTION.
    ex_thresholds : Dictionary
        DESCRIPTION.
    ex_correlation_way_dict : Dictionary, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    corr_dict : Dictionary
        DESCRIPTION.

    '''
    #Make sure one only uses the train data
    train_data = data.iloc[:start,:]
    ex_train = ex.iloc[:start,:]
    
    #Initialize a dataframe to save all the correlations per time series
    corr_df = pd.DataFrame(index = train_data.columns)
    
    #For some variables it might work better to calculate the correlation with the differenced time series
    #One can think of special events, such as promotions or national holidays
    if ex_correlation_way_dict != {}:
        #Determine it per external variable
        for col in ex_train.columns:
            ex_serie = ex_train[col]
            if ex_correlation_way_dict[col] == 'differenced':
                #Difference both the train data and the external variable
                train_data_col = train_data.diff(1).iloc[1:,:]
                ex_serie = ex_serie.diff(1).iloc[1:,:]
                
            else:
                #Nothing changes and the correlation is calculated between the real variables
                train_data_col = train_data
                
            #Determine the correlation between the whole data set and the external variable
            corr_df[col] = train_data_col.apply(lambda s: abs(ex_serie.corr(s)), axis = 0)                
            
    #Else perform the same procedure on the real data
    else:
        #Determine it per external variable
        for col in ex_train.columns:
            ex_serie = ex_train[col]
                
            #Determine the correlation between the whole data set and the external variable
            corr_df[col] = train_data.apply(lambda s: abs(ex_serie.corr(s)), axis = 0)           
    
    
    #Compare the variables with the thresholds
    corr_dict = {}
    
    #Create a list of the threshold dictionary, such that it better comparable(order of external variables is similair as above)
    corr_thresholds = [ex_thresholds[col] for col in ex_train.columns]
    
    #Compare the correlations for every colum in the data set
    for col in train_data.columns:
        #Check for every row in the corr df it is > correlation thresholds
        corr_col = corr_df.loc[col,:] > corr_thresholds
        
        #Get the names from the external variables for which the correlation > thresholds
        corr_dict[col] = list(corr_col[corr_col == True].index)

    #Return the correlation dictionary
    return corr_dict    
    
    



