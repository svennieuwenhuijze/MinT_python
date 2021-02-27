# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:26:16 2020

@author: svenn
"""

# %%Import required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm

'''This module contains an estimation and forecasting function for the ARX model,
    which is given by yt = l_1 f_1 + .. + l_r f_r +a_1 y_(t-1) + ... + a_p y_(t-p) + b x_(1,t) ... b_q x_(q,t).
    Moreover it is optional to preprocess the data, using the preprocess model
    '''

#%% ARX class

class FAVARX():
    
    #Define the inital state of the ARX model
    def __init__(self, Y):
        '''Initiate the arx model, using a time series (data set)
        
            Y: nump.array or pandas.DataFrame
               All time series'''
                
        #Data attributes
        self.data = Y
        self.ex = []
        
        #Parameter attributes
        self.params = []
        self.lags = {}
        self.ex_struct = {}
        self.common_factors = None
        self.common_factors_params = None
        
        #Preprocess attributes
        self.log = True
        self.stationarity = True
        self.prepare = True
        self.preprocess_info = pd.DataFrame()
        self.prepared_data = pd.DataFrame()
        self.logged_data = pd.DataFrame()
    
        
    #Function to fit the ARX model for every time series in Y
    def fit(self, ex = None, lags = None, ex_struct = None, ex_thresholds = None, log = True, stationarity = True, prepare = True):
        '''
        Function which fits the ARX model on every time series of the data set Y

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        lags : TYPE, optional
            DESCRIPTION. The default is {}.
        ex_struct : TYPE, optional
            DESCRIPTION. The default is {}.
        ex_thresholds: dictionary
            Contains for every external variable the minimal correlation coefficient.
            If correlation between y_t and x_t > threshold, x_t influences y_t and is 
            included in to the model
        Returns
        -------
        params : numpy.array
                 Contains the weights of the fitting procedure for every variable. 
                 Weights for the model of time series i can be found in the ith row
        lags: python dictionary
              #optimal number of lags obtained by fitting procedure if not predefined
        ex_struct: python dictionary
                   For every time series i in Y it contains the column names of the 
                   external variables which influence i, based on preprocess
        '''
        #Import the libraries specifically required for this function       
        from prepare_functions import prep_data, AR_struct, ex_corr_struct
        
        #Start with validating all inputs (e.g. if they have the right types, shapes etc..)
        
        #If one added external variables to the model determine which variable influences which time series
        if ex is not None:
            self.ex = ex
            #If ex_structure is not predefined find out which external variable influences which time series
            if ex_struct is None:
                #If the thresholds are not defined, set the 
                if ex_thresholds is None:
                    ex_thresholds = pd.Series([0] * ex.shape[1], index = ex.columns).to_dict()
                self.ex_struct = ex_corr_struct(self.data, self.data.shape[0], ex, ex_thresholds)
            else:
                self.ex_struct = ex_struct
                
        #If desired prepare the data with the functions in the prepare class
        if prepare:
            #Perform a standard prepare version, if desired, one can perform the preprocess outside this fitting function
            prep_dict = prep_data(self.data, self.data.shape[0], stationair=stationarity, log=log, ex = self.ex, ex_struct = self.ex_struct, ex_control = None)
            self.prepared_data = prep_dict['Prepared data']
            self.logged_data = prep_dict['Logged data']
            self.preprocess_info = prep_dict['Preprocess info']
        
        #Else set it both equal to the original data
        else:
            self.prepared_data, self.logged_data = self.data, self.data
        
        #Update the attributes
        self.prepare, self.log, self.stationarity = prepare, log, stationarity
        
        #If lags is not predefined determine them, else use the predefined lags
        if lags is None:
            self.lags = AR_struct(self.prepared_data, self.prepared_data.shape[0], maxlag = 12, ic = 'aic')
        else:
            self.lags = lags
        
        #If prepare is true drop the first observation, due to differencing
        if prepare & stationarity:
            self.prepared_data = self.prepared_data.iloc[1:,:]
            if ex is not None:
                ex = ex.iloc[1:,:]
                
        #Also get the maximum number of lags
        max_lags = max(self.lags.values())

        #======================================================================
        #Convert the data to numpy array for multiplication
        Y_np = self.prepared_data.values
        n_series = Y_np.shape[1]
        
        #Define the matrix Y, containing all dependent variables (time series)
        Y_dep = Y_np[max_lags:,:].T        
        
        #Create the lag matrix by storing all (max)lags below each other ((n_lags * N) X T)
        Y_indep = np.ones((1,Y_dep.shape[1])) #Start by generating the constant variable
        #Now add the matrix of every lag below each other
        for p in range(1,max_lags+1):
            Y_1 = Y_np[(max_lags-p):-p,:].T
            Y_indep = np.vstack([Y_indep, Y_1])

        #Add the external variables
        if ex is not None:
            Y_indep = np.vstack([Y_indep, ex.values[max_lags:,:].T])            
        
        #======================================================================
        #Convert the dictionaries based on the column names to dictionaries based on position
        #such that they can be applied to pandas dataframes and numpy arrays        
        
        #if there is an external variable present, perform some preprocessing
        if ex is not None:
            #Convert the ex structure dict into an ex numeric structure dict --> easier to work with numpy arrays
            const = 1+ max_lags * n_series #constant consist out of all the lags and the 1 for the constant term in the linear eq
            
            #Create a dict between the column name and its numerical position
            ex_pos_dict = {}
            for col in ex.columns:
                ex_pos_dict[col] = ex.columns.get_loc(col)
                
            #Use this dict to transfer the ex_struct to numeric format --> where the numbers represent the row in the matrix
            ex_position_num = {}
            for i, col in enumerate(self.data.columns):
                ex_position_num[i] = [const + ex_pos_dict[x] for x in self.ex_struct[col]]
        else:
            #Fill the dictionary with empyt values
            ex_position_num = {}
            for i in range(n_series):
                ex_position_num[i] = []
                
        #Create list of at which positions the regressors in D are for every time series. 
        lags_position_num = {}
        for i, col in enumerate(self.data.columns):
            lags_position_num[i] = [0] #The constant is always a regressor, and located at position 0
            
            #Now add all the positions of the lags for the specific column 
            for j in range(self.lags[col]):
                lags_position_num[i].append((j*n_series)+i+1)
            
        #======================================================================
        #Fit every ARX model using linear regression and save the parameters in the coefficient matrix self.params
        self.params = np.zeros((n_series, Y_indep.shape[0]))
        
        for i in range(n_series):
            #Define exogenoues variables
            X = Y_indep[lags_position_num[i] + ex_position_num[i],:].T
            #Fit the model
            fit = sm.OLS(Y_dep[i,:].T, X).fit()
            #Save the parameters
            self.params[i, lags_position_num[i] + ex_position_num[i]] = fit.params
            
        fittedvalues = pd.DataFrame(self.params.dot(Y_indep).T, columns = self.data.columns)            
        
        #If the data is prepared invert the preprocess to obtain the fittedvalues to the original data
        if prepare:
            prep_fittedvalues = fittedvalues.copy()
            #Define the differenced time series
            diff_series = list(self.preprocess_info.loc[self.preprocess_info['differenced'] == True, :].index) 
            
            #Invert the differencing by adding the differenced fitted values to the original log values (important to get the right points)
            fittedvalues[diff_series] = self.logged_data[diff_series].iloc[max_lags:-1,:] + prep_fittedvalues[diff_series]
            
            #Invert the log operator and obtain the fitted values for the original data
            fittedvalues = np.exp(fittedvalues)            
        else:
           #Prepared fitted values are similair to the fitted values
           prep_fittedvalues = fittedvalues 

        return ARX_fit(self.data, fittedvalues, prep_fittedvalues, self.params, self.lags, self.ex_struct, self.prepare, self.prepared_data, self.logged_data, self.preprocess_info)

# %% Define the class which defines a fitted ARX model
class ARX_fit():

    def __init__(self, data, fit, prep_fit, params, lags = None, ex_struct = None, prepare = True, prepared_data = None, logged_data = None, prepare_info = None):
        '''Initialize the fitted object'''
        
        #Data attributes
        self.data = data
        self.prep_fittedvalues = prep_fit        
        self.fittedvalues = fit
        
        #params attributes
        self.params = params
        self.lags = lags
        self.ex_struct = ex_struct
        
        #preprocess attributes
        self.prepare = prepare
        self.logged_data = logged_data
        self.prepared_data = prepared_data
        self.preprocess_info = prepare_info
        
        
    def forecast(self, h, ex_test = None):
        '''
        Parameters
        ----------
        h : int
            The number of periods to be forecasted ahead.
        ex_test : pandas.DataFrame
            values of external variables in testing period.

        Returns
        -------
        fc: pd.DataFrame()
            h step ahead forecasts for all time series.

        '''
        #Create the regressing matrix based on the max lags
        Y = self.prepared_data.values; max_lags = max(self.lags.values())
        
        #Start with a one for the constant
        Y_indep = np.ones((1,1))
        
        #Add all the max_lags
        for lag in range(max_lags):
            Y_indep = np.vstack([Y_indep, Y[-lag,:].T.reshape(-1,1)])
        
        #Now also add the external variables if necessary
        if ex_test is not None:
            Y_indep = np.vstack([Y_indep, ex_test.values[0,:].T.reshape(-1,1)])
        
        #Save the forecasts in a numpy array
        fc = np.zeros((h, self.prepared_data.shape[1]))
        
        #Forecast for the range of the forecast horizon h
        for i in range(h):
            if i == 0:
                fc[i,:] = self.params.dot(Y_indep).T
            else:
                #Redefine the regressing vector (new lags and external variables(if present))
                if ex_test is not None:
                    Y_indep = np.vstack([1, fc[i-1,:].T.reshape(-1,1), Y_indep[1:-(self.prepared_data.shape[1]+ex_test.shape[1])], ex_test[i,:].values.reshape(-1,1)])
                else:
                    Y_indep = np.vstack([1, fc[i-1,:].T.reshape(-1,1), Y_indep[1:-self.prepared_data.shape[1]]])
                    
                #Forecast further away
                fc[i,:] = self.params.dot(Y_indep).T

        fc = pd.DataFrame(fc, columns = self.data.columns)
        
        #If the data is prepared invert the preprocess
        if self.prepare:
            prep_fc = fc.copy()
            
            #Define the differenced time series
            diff_series = list(self.preprocess_info.loc[self.preprocess_info['differenced'] == True, :].index) 
            
            #Start with rolling back the differences
            fc[diff_series] = self.logged_data[diff_series].iloc[-1,:] + prep_fc[diff_series].cumsum()
            
            #Now invert the log operation
            log_cols = list(self.preprocess_info.loc[self.preprocess_info['log'] == True, :].index)
            fc[log_cols] = np.exp(fc[log_cols])
        else:
            prep_fc = fc.copy
            
        #Return an ARX forecast object
        return ARX_fc(fc, self.data, prep_fc, self.prepared_data)
            
# %% Forecast class

class ARX_fc():
    '''Only to initialize a forecast object which, such that the attributes are easily accesible '''
    def __init__(self, fc, data, prep_fc = pd.DataFrame(), prep_data = pd.DataFrame()):
        
        #Forecast objects
        self.forecasts = fc
        self.forecasts_prep_data = prep_fc
        
        #Data attributes
        self.data = data
        self.prepared_data = prep_data
        self.data = data
        self.prep_data = prep_data
