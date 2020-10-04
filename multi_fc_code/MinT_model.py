# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:27:44 2020

@author: svenn
"""

#Import required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from more_itertools import unique_everseen
import statsmodels.stats.moment_helpers as moment_helpers

#Import other file
import multi_fc_code.ARX_model as ARX_model

#%% Initiate a hierarichal class

class MinT:
    
    def __init__(self, hierarchical_structure, bottom_level, hierarchical_levels):
        '''
        Parameters
        ----------
        hierarchical_structure : pandas.DataFrame
            hierarchical levels as index and hierarchical structure as values
            (see example below for clarification). Make sure the lowest level
            is the first row of the list and total at the last row. 
            The whole structure is based on the alphabetic order of the lowest level
        bottom_level : list
            Contains the one or multiple levels which form the bottom level, sorted according
            to the hierarchical structure
        hierarchical_levels: list of lists
            All levels you want to use and create forecasts on. 
            (See example below for clarification). Especially convenient when
            you want to generate a grouped structure. Make sure this follows the same
            order as the actuals and forecast inputs. Moreover all the levels, which
            are in the bottom level should be in here.
        
        Returns
        -------
        None.

        '''
        #Structure parameters
        self.hierarchical_structure = hierarchical_structure
        self.bottom_level = bottom_level
        self.hierarchical_levels = hierarchical_levels
        self.hierarchical_order = []
        
        #Fit parameters
        self.S = None
        self.P = None
        self.cov_est = None
        self.shrink_param = None
        self.base_model = None
        
        

    def smatrix(self):
        '''
        Function which returns the summation matrix given a certain hierarchical
        structure
    
        Parameters
        ----------
        None
    
        Returns
        -------
        S : numpy.array
        Representing the summation matrix.
    
        '''
        
        #Obtain the number of bottom time series
        num_bts = self.hierarchical_structure.shape[1]
        
        #Obtain the number of total time series
        num_ts = 1 # 1 for the total
        for level in self.hierarchical_levels:
            num_ts += self.hierarchical_structure.loc[level,:].T.drop_duplicates().shape[0]
        
        #Initiate the summation matrix
        S = np.zeros((num_ts,num_bts))
        
        #Fill the top row
        S[0,:] = 1
        
        #Initaite a variable which keeps track of the position of time series
        S_position = 1
        
        #Fill the intermediate levels (all levels except the bottom level)
        for level in self.hierarchical_levels[:-1]:
            #Get the structure of the levels
            level_structure = self.hierarchical_structure.loc[level,:]
            
            #Get an ordered list of time series in the considered level
            if len(level) > 1:
                #Firstly make tuples from the level info if there are more than one
                level_names = level_structure.apply(tuple, axis=0).tolist()
                
                #Create an ordered time series list
                level_ordered_ts = list(unique_everseen(list(level_names)))
            else:
                #Firstly make tuples from the level info if there are more than one
                level_names = level_structure.values[0]
                
                #Get the list of unique everseen time series (an ordered time series list)
                level_ordered_ts = list(unique_everseen(list(level_names)))
         
            #For every time series find the location of the bottom time series --> locations where summation matrix = 1
            for ts in level_ordered_ts:
                #Get the position for the ts
                ts_s_positions = [i for i,e in enumerate(level_names) if e == ts]
                #Set those postions equal to one
                S[S_position, ts_s_positions] = 1
                #Update the s position
                S_position += 1
            
        #Fill the bottom rows (equal to the identity matrix)
        S[-num_bts:,:] = np.identity(num_bts)
    
        return S

    def shrinkage_est(self, res, target):
        '''
        Estimate the covariance matrix, using the shrinkage estimator

        Parameters
        ----------
        res : pandas.DataFrame or numpy.array
            DESCRIPTION.
        target : pandas.DataFrame or numpy.array
            DESCRIPTION.

        Returns
        -------
        shrink_cov : numpy.array
            Shrinkage covariance estimator.
        lamb : double
            Shrinkage parameter.
        '''
        
        #Make sure res and target are np arrays
        res = np.array(res); target = np.array(target)
    
        #Get the parameters of the residuals matrix
        n = res.shape[0]
    
        #Get the sample correlation and covariance matrix of the residuals
        covm = res.T.dot(res) / n
        corm = moment_helpers.cov2corr(covm)
    
        #Give the residuals equal standard deviation (=1)
        res_scaled = res / np.diag(covm) ** 0.5
    
        #Define the shrinking intensity lambda
        v = (1/(n*(n-1))) * (((res_scaled**2).T.dot(res_scaled**2))  - (1/n) * ( ((res_scaled).T.dot(res_scaled)) **2))
        np.fill_diagonal(v, 0)
    
        corapn = moment_helpers.cov2corr(target)
        d = (corm - corapn) ** 2
    
        lamb = sum(sum(v))/sum(sum(d))
        lamb = max(min(lamb,1),0)
    
        #Define the shrinkage estimator
        shrink_cov = lamb * target + (1-lamb) * covm
    
        return shrink_cov, lamb


    def fit_py(self, actuals, fittedvalues = None):
        '''
        Estimate the MinT parameters without the rpy2 dependency

        Returns
        -------
        None.

        '''
        #get the summation matrix
        self.S = self.smatrix()

        #If there are no fittedvalues determine them
        if fittedvalues is None:
            #Initiate the model
            ARX = ARX_model.ARX(actuals)
            #Fit the model
            self.base_model = ARX.fit()
            #Get the fittedvalues from the model
            fittedvalues = self.base_model.fittedvalues           
        
        #Order the fittedvalues and the actuals similar to the summation matrix
        if self.hierarchical_order == []:
            #However, only if this has not been done before
            for level in self.hierarchical_levels:
                if len(level) > 1:
                    #Firstly make tuples from the level info if there are more than one
                    level_names = self.hierarchical_structure.loc[level,:].apply(tuple, axis=0).tolist()
                    level_order = (list(unique_everseen(list(level_names))))
                    [self.hierarchical_order.append(x) for x in level_order]
                else:
                    #append the column names to the hierarchical order following the structure of the hierarchy object
                    level_order = list(unique_everseen(list(self.hierarchical_structure.loc[level,:].values[0])))
                    [self.hierarchical_order.append(x) for x in level_order]
            
            #Add the total colum as well
            self.hierarchical_order.insert(0, 'Total')
        
        #Reorder the data
        actuals = actuals[self.hierarchical_order]
        fittedvalues = fittedvalues[self.hierarchical_order]        
        
        #Get the in-sample residuals and calculate the shrinkage estimator of the covariance matrix
        residuals = pd.DataFrame(actuals.iloc[(actuals.shape[0] - fittedvalues.shape[0]):,:].values - fittedvalues.values, columns = fittedvalues.columns)        
        
        #First of all define the target estimator
        n = residuals.shape[0]
        target = np.diag(np.diag((residuals.T.values.dot(residuals.values)/ n)))
        
        #Calculate the shrinkage covariance estimator
        self.cov_est, self.shrink_param = self.shrinkage_est(residuals, target)
               
        #Lastly determine the optimal reconciliation matrix P (analytical solution of MinT)
        self.P = np.linalg.inv(self.S.T.dot(np.linalg.inv(self.cov_est).dot(self.S))).dot(self.S.T.dot(np.linalg.inv(self.cov_est)))
        
    def forecast_py(self, h, base_forecasts = None):
        '''
        Generate the reconcilled forecasts with MinT
    
        Inputs
        ------
        h : integer
        Forecast horizon
        base_forecasts : pandas.DataFrame 
        Contains the independently created base forecasts for the whole hierarchy
        (optional, default = None)

        Returns
        -------
        Reconcilled forecast for the whole hierarchy.

        '''
        #If there are no base forecasts calculate them with the model initiated during fitting procedure
        if base_forecasts is None:
            #Use the model fitted during the fit process
            base_model = self.base_model
            #Forecast
            ARX_forecasts = base_model.forecast(h); base_forecasts = ARX_forecasts.forecasts
        
        #Make sure the base forecasts are in the correct order
        base_forecasts = base_forecasts[self.hierarchical_order]
        
        #Calculate the MinT forecasts
        MinT_fc = self.S.dot(self.P.dot(base_forecasts.values.T)).T
        
        #Convert the forecasts into a dataframe
        MinT_fc = pd.DataFrame(MinT_fc, columns = base_forecasts.columns)
        
        #Return the forecasts
        return MinT_fc
        
    def fit_forecast_rpy2(self, actuals, h, base_forecasts = None, fittedvalues = None):
        '''
        Fit and generate reconcilled forecasts. Based on the base_forecasts if present, 
        otherwise the independent nonreconcilled base forecasts will be generated by ARX

        Parameters
        ----------
        actuals : pandas.DataFrame
            Contains all actuals. Time periods as index and units as columns.
            Does not need to be ordered perfectly.
        forecasts : pandas.DataFrame
            Contains all fitted values + h forecasts. Time periods as index and units as columns.
            Does not need to be ordered perfectly
        h : integer
            Forecast horizon
        Returns
        -------
        forecasts : pandas.DataFrame
            Contains all h-step ahead reconcilled forecasts for the whole hierarchy.

        '''
        #######################################################################
        #Import libraries and install the hts library in the r instance
        from rpy2.robjects.packages import importr
        from rpy2.robjects import r, pandas2ri
        pandas2ri.activate()
        #utils = importr('utils')
        #utils.chooseCRANmirror(ind=1)
        #utils.install_packages('hts')
        hts = importr('hts')
        
        
        #######################################################################
        #Initiate the hts object based on the hierarchical structure, in order
        #to do so we need the actuals of the bottom level of the hierarchy and 
        #the hierarchical structure
        
        #Get the bottom column names (tuples of the hierarchical structure columns)
        bottom_columns = self.hierarchical_structure.loc[self.bottom_level,:]
        if len(self.bottom_level) > 1:
            bottom_columns = bottom_columns.apply(tuple, axis=0).tolist()
        else:
            bottom_columns = bottom_columns.values[0].tolist()
        
        #Now get the bottom actuals
        bottom_actuals = actuals[bottom_columns]
        
        #Remove the bottom level from the hierarchical structure, because the hts package automatically creates it
        if len(self.bottom_level) == 1:
            hierarchical_structure = self.hierarchical_structure.copy()
            hierarchical_structure = hierarchical_structure.drop(self.bottom_level, axis = 0)
        else:
            hierarchical_structure = self.hierarchical_structure
            
        #Convert the hierarchical structure to a R object
        hierarchical_structure_r = pandas2ri.py2ri(hierarchical_structure)
        hierarchical_structure_r = r['as.matrix'](hierarchical_structure_r)
        
        #Create the hierarchical structure with rpy2
        hierarchical_object = hts.gts(pandas2ri.py2ri(bottom_actuals), groups = hierarchical_structure_r)
        
        #Get the summation matrix
        self.S = pandas2ri.ri2py(hts.smatrix(hierarchical_object))
        
        #######################################################################
        #Generate the nonreconcilled forecasts with the ARX model (if this has not been done yet)
        if base_forecasts is None:
            #Initiate the model
            ARX = ARX_model.ARX(actuals)
            #Fit the model
            fit = ARX.fit()
            #Forecast
            ARX_forecasts = fit.forecast(h); base_forecasts = ARX_forecasts.forecasts
            #Also keep the fittedvalues for the residuals
            fittedvalues = fit.fittedvalues
       
        #######################################################################
        #order the actuals and forecast such that they follow the hierarchical object
        if self.hierarchical_order == []:
            #However, only if this has not been done before
            for level in self.hierarchical_levels:
                if len(level) > 1:
                    #Firstly make tuples from the level info if there are more than one
                    level_names = self.hierarchical_structure.loc[level,:].apply(tuple, axis=0).tolist()
                    level_order = (list(unique_everseen(list(level_names))))
                    [self.hierarchical_order.append(x) for x in level_order]
                else:
                    #append the column names to the hierarchical order following the structure of the hierarchy object
                    level_order = list(unique_everseen(list(self.hierarchical_structure.loc[level,:].values[0])))
                    [self.hierarchical_order.append(x) for x in level_order]
            
            #Add the total colum as well
            self.hierarchical_order.insert(0, 'Total')
        
        #Reorder the data
        actuals = actuals[self.hierarchical_order]
        base_forecasts = base_forecasts[self.hierarchical_order]

        #######################################################################
        #Generate the reconcilled forecasts with MinT
        
        #Get the insample residuals, note that due to the fitting procedure you presumably lost some observations
        residuals = pd.DataFrame(actuals.iloc[(actuals.shape[0] - fittedvalues.shape[0]):,:].values - fittedvalues.values, columns = fittedvalues.columns)

        #Convert the objects to r_objects
        residuals_r = r['as.matrix'](residuals)
        base_forecasts_r = r['as.matrix'](base_forecasts)

        #Generate the MinT forecastss
        MinT_fc = hts.MinT(fcasts = base_forecasts_r, groups = r['get_groups'](hierarchical_object), residual = residuals_r, covariance = 'shr', algorithms = 'lu')
        
        #Obtain the MinT forecasts for the bottom level
        MinT_fc = pandas2ri.ri2py(MinT_fc[0])
        
        #Now obtain them for the whole hierarchy and store them in a dataframe
        MinT_fc = pd.DataFrame(self.S.dot(MinT_fc.T).T, columns = actuals.columns)
        
        return MinT_fc


