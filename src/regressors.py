#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:14:11 2019

@author: Niccolò Mora
"""

import numpy as np
import pandas as pd
from sklearn import linear_model as sklm
import statsmodels.api as sm
from scipy import stats


class GlmOlsRegressor:
    """
    Class for wrapping GLM fitting and prediction. Default uses OLS regression,
    i.e. Gaussian family.
    """
    def __init__(self):
        self.mdl = None
        self.fitres = None
        self.family = sm.families.Gaussian()
        return
    
    def _validate_X_y(self,X,y):
        """
        Validate model input and dependent variable.
        Args:
            -X (2D array): matrix of regressors
            -y (2D array): desired output
        """
        if len(y.shape)!=2:
            raise ValueError('expected y to be 2D, found {}'.format(len(y.shape)))
        if X is None:
            X = np.ones(y.shape[0]).reshape(-1,1)
        elif len(X.shape)!=2:
            raise ValueError('expected X to be 2D, found {}'.format(len(X.shape)))
        return (X,y)
    
    def gross_outliers_y(self,y,minval=-np.inf,maxval=np.inf,iqr_factor=3.0):
        """
        Find extreme outliers from data.
        Args:
            -y: data to clean
            -minval,maxval: minimum/maximum allowable values for inliers
            -iqr_factor: multiplicative factor for the Inter-Quartile Range
        Returns:
            -inliers: logical vector for addressing inliers
        """
        yloc = y.ravel()
        inlierminmax = np.logical_and(yloc<=maxval,yloc>=minval)
        pct25,pct75 = np.percentile(yloc[inlierminmax],[25,75])
        upper = pct75+iqr_factor*(pct75-pct25)
        lower = pct25-iqr_factor*(pct75-pct25)
        inliers = np.logical_and(np.logical_and(yloc<=upper,yloc>=lower),
                                 inlierminmax)
        return inliers
    
    def fit(self,X,y,discard_outliers=False,minval=-np.inf,maxval=np.inf,
            iqr_factor=3.0):
        """
        Fit model.
        Args:
            -X (2D array): matrix of regressors
            -y (2D array): desired output
            -discard_outliers (boolean): apply gross_outliers_y() and discard
                gross outliers. Defaults to False
            -minval,maxval: minimum/maximum allowable values for inliers.
            -iqr_factor: multiplicative factor for the Inter-Quartile Range
        """
        just_intercept = X is None
        X,y = self._validate_X_y(X=X,y=y)
        if not just_intercept:
            X = sm.add_constant(X,prepend=False)
        if discard_outliers:
            inliers = self.gross_outliers_y(y,minval=minval,maxval=maxval,iqr_factor=iqr_factor)
        else:
            inliers = np.array(y.shape[0]*[True])
        X = X[inliers,:]
        y = y[inliers,:]
        self.mdl = sm.GLM(y, X, family=self.family)
        self.fitres = self.mdl.fit()
        return (self.fitres,inliers)
    
    def predict(self,X):
        return self.mdl.predict(X)




class GlmPoissonRegressor(GlmOlsRegressor):
    """
    Class for wrapping Poisson GLM fitting and prediction.
    """
    def __init__(self):
        self.mdl = None
        self.fitres = None
        self.family = sm.families.Poisson()
        return
    
    def gross_outliers_y(self,y,minval=-np.inf,maxval=np.inf,iqr_factor=2.5):
        """
        Find extreme outliers from data. Log-count based IQR filtering
        Args:
            -y: data to clean
            -minval,maxval: minimum/maximum allowable values for inliers
            -iqr_factor: multiplicative factor for the Inter-Quartile Range
        Returns:
            -inliers: logical vector for addressing inliers
        """
        yloc = y.ravel()
        inlierminmax = np.logical_and(yloc<=maxval,yloc>=minval)
        yloc = np.log(1+yloc)
        pct25,pct75 = np.percentile(yloc[inlierminmax],[25,75])
        upper = pct75+iqr_factor*(pct75-pct25)
        lower = pct25-iqr_factor*(pct75-pct25)
        inliers = np.logical_and(np.logical_and(yloc<=upper,yloc>=lower),
                                 inlierminmax)
        return inliers




class GlmNegBinRegressor(GlmOlsRegressor):
    """
    Class for wrapping Negative Binomial GLM fitting and prediction.
    """
    def __init__(self):
        self.mdl = None
        self.fitres = None
        self.family = sm.families.NegativeBinomial()
        return




if __name__ == '__main__':
    print('Test Area')
    import matplotlib.pyplot as plt
    np.random.seed(2019)
    
    npoints = 500
    
    trend_exp = 1.5
    pois_lambda = 5.5
    #negbin_var = pois_lambda*1.4
    
    #negbin_n = pois_lambda**2/(negbin_var-pois_lambda)
    #negbin_p = pois_lambda/negbin_var
    
    Xgenerator_poisson = pois_lambda*np.exp(np.linspace(0,trend_exp,num=npoints))
    negbin_var = Xgenerator_poisson*1.4
    Xgenerator_negbin_n = Xgenerator_poisson**2/(negbin_var-Xgenerator_poisson)
    Xgenerator_negbin_p = Xgenerator_poisson/negbin_var
    
    y_poisson = np.array([stats.poisson.rvs(x) for x in Xgenerator_poisson]).reshape(-1,1)
    y_negbin = np.array([stats.nbinom.rvs(xn,xp) for xn,xp in \
                         zip(Xgenerator_negbin_n,Xgenerator_negbin_p)]).reshape(-1,1)
    
    X = np.linspace(0,1,npoints).reshape(-1,1)
    #X=None
    y = y_negbin
    
    
    mdl_poisson = GlmPoissonRegressor()
    res_poisson,inliers_poisson = mdl_poisson.fit(X,y,discard_outliers=True)
    print(res_poisson.summary())
    print('Outliers: {:d}'.format(npoints-np.sum(inliers_poisson)))
    print('x1: {:.3f} \nconst: {:.3f}'.format(np.exp(res_poisson.params[0]),
                                              np.exp(res_poisson.params[1])))
    
    mdl_negbin = GlmNegBinRegressor()
    res_negbin,inliers_negbin = mdl_negbin.fit(X,y,discard_outliers=True)
    print(res_negbin.summary())
    print('Outliers: {:d}'.format(npoints-np.sum(inliers_negbin)))
    print('x1: {:.3f} \nconst: {:.3f}'.format(np.exp(res_negbin.params[0]),
                                              np.exp(res_negbin.params[1])))
    
    print('\nTrue:\nx1: {:.3f} \nconst: {:.3f}'.format(np.exp(trend_exp),
                                                  pois_lambda))
    fig,ax = plt.subplots()
    ax.scatter(np.arange(npoints),y)
    
	
    
    
    
    