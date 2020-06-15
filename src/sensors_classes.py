#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:58:08 2019

@author: Niccolò Mora
"""

import pandas as pd
import numpy as np
from itertools import groupby


class GENERICBINsensor():
    def __init__(self, name,
                 fullres_minutes=1,
                 timezone_adjust=None):
        """
        Generic class for binary sensor
        Args:
            -name (str): name of the sensor will be the data column name as well
            -fullres_minutes (int, optional): time resolution, in minutes, of
                the full resolution signature matrix. Defaults to 1.
            -timezone_adjust (str, optional): timezone to apply, if needed. 
                Defaults to None
        Returns:
            no values returned
        """
        self.name = name
        self.fullres_minutes = fullres_minutes
        self.timezone_adjust = timezone_adjust
        return
    
    def _import_data(self, data, replace_minus1=True):
        """
        Local function for importing data and setting appropriate timezone
        Args:
            -data (pandas.TimeSeries): timeseries of raw sensor data
            -replace_minus1 (bool, optional): replace -1 values with a 0.
                Defaults to True
        Returns:
            -dftmp (pandas.DataFrame): a dataframe with appropriate timezone 
                and sorted index
        """
        dftmp = pd.DataFrame(data.values,index=data.index,columns=[self.name])
        dftmp.sort_index(inplace=True)
        if self.timezone_adjust:
            dftmp = dftmp.tz_localize('UTC').tz_convert(self.timezone_adjust).tz_localize(None)
        if replace_minus1:
            dftmp[self.name].replace(to_replace=-1,value=0)
        return dftmp
    
    def set_battery_data(self, data):
        """
        Create a battery DataFrame
        Args:
            -data (pandas.DataFrame): dataframe featuring a 'value' column and 
                a 'timestamp' index
        Returns:
            No values returned. Data will be stored in self.batterydata.
        """
        self.batterydata = self._import_data(data,replace_minus1=False)
        return
    
    def import_fullres_sigmatr(self,sigmatr):
        """
        Import a previously computed signature matrix. Time resolution will be
        resampled to meet the fullres_minutes parameter of the constructor.
        Args:
            -sigmatr (pandas.TimeSeries): a pre-computed binary signature matrix
        Returns:
            No values returned. Full resolution matrix will be stored
            internally in self.fullres_m.
        """
        resample_rule = '{}T'.format(self.fullres_minutes)
        resampler = sigmatr.resample(resample_rule)
        self.fullres_m = resampler.max()
        return
    
    def compute_fullres_sigmatr(self, data, replace_minus1=True):
        """
        Create a signature matrix from an input dataframe
        Args:
            -data (pandas.DataFrame): dataframe featuring a 'value' column and a 
                'timestamp' index
            -replace_minus1 (bool, optional): replace -1 values with a 0.
                Defaults to True
        Returns:
            No values returned. Full resolution matrix will be stored
            internally in self.fullres_m.
        """
        df = self._import_data(data,replace_minus1=replace_minus1)
        d_begin,d_end = df.index.date[[0,-1]]
        t_begin = pd.Timestamp(year=d_begin.year,month=d_begin.month,day=d_begin.day,
                               hour=0,minute=0,second=1)
        t_end = pd.Timestamp(year=d_end.year,month=d_end.month,day=d_end.day,
                               hour=23,minute=59,second=59)
        df = pd.concat([df,
                        pd.DataFrame(index=[t_begin],columns=[self.name]).fillna(0),
                        pd.DataFrame(index=[t_end],columns=[self.name]).fillna(0) ],
                       axis=0)
        df.sort_index(inplace=True)
        df = df.clip(0,1)
        df['filled'] = 0
        resample_rule = '{}T'.format(self.fullres_minutes)
        resampler = df.resample(resample_rule)
        #resample by keeping activations and propagating those values to na
        sens_df_max = resampler.max()
        sens_df_max['filled'] = sens_df_max['filled'].fillna(1)
        sens_df_max = sens_df_max.fillna(method='ffill')
        sens_df_max = sens_df_max.fillna(0)
        #resample by keeping last values and propagating those values to na
        sens_df_last = resampler.last()
        sens_df_last['filled'] = sens_df_last['filled'].fillna(1)
        sens_df_last = sens_df_last.fillna(method='ffill')
        sens_df_last = sens_df_last.fillna(0)
        #combine activations: basically, when missing values were filled,
        #the right one is sens_df_last. In order not to miss any activation,
        #e.g. when the last val is 0 but there was a 1 within the same minute,
        #copy the non-filled data from sens_df_max
        vals_max = sens_df_max[self.name].values
        vals_last = sens_df_last[self.name].values
        new_vals = np.where(sens_df_max.filled==1,vals_last,vals_max)
        self.fullres_m = pd.DataFrame(index=sens_df_max.index,
                                 data=new_vals,
                                 columns=[self.name])
        
        return
    
    def get_active_ts(self, resolution_u=30,
                      mode='integral',
                      binary_minduration_u=1):
        """
        Get an activation matrix, in a resampled resolution. Provides various
        output options: {'integral','normalized','binary'}
        Args:
            -resolution_u (int): resolution of the output dataframe, in 
                fullres_minutes units. Defaults to 30.
            -mode (string): mode for output active matrix. Can be either:
                *'integral': activations of the fullres_m are summed
                *'normalized': activations of the fullres_m are summed and 
                    normalized to 1 in each sample
                *'binary': activations are binarized, i.e. set to 1 if at least
                    binary_minduration_u instants are set to 1, otherwise it is
                    set to 0
            -binary_minduration_u (int): minimum active fullres_minutes units
                to consider a sensor active.
        Returns:
            -active_ts (pandas.TimeSeries): a time series with computed 
                activation scores
        """
        resample_rule = '{}T'.format(self.fullres_minutes*resolution_u)
        resampler = self.fullres_m.resample(resample_rule)
        active_df = resampler.sum()
        if mode=='integral':
            return active_df[self.name].copy()
        elif mode=='normalized':
            return active_df[self.name].copy()/resolution_u
        elif mode=='binary':
            binarized = active_df[self.name].apply(lambda x: 0 if x<binary_minduration_u else 1)
            return binarized
        else:
            raise ValueError('Invalid mode parameter')
        
    def get_active_lookback_ts(self,resolution_u=30,
                               lookback_u=60):
        """
        Compute active time units within the last lookback_u, with an update
        resolution of resolution_u
        Args:
            -resolution_u (int): resolution of the output dataframe, in 
                fullres_minutes units. Defaults to 30.
            -lookback_u (int): how amny fullres_minutes units to look back for 
                discovering activity bouts. Defaults to 60.
        Returns:
            -active_lookback_ts (pandas.TimeSeries): a time series with the 
                active time within last lookback units 
                (base unit = fullres_minutes)
        """
        integrale_df = self.fullres_m.rolling(window=lookback_u).sum()
        resample_rule = '{}T'.format(self.fullres_minutes*resolution_u)
        active_lookback_df = integrale_df.resample(resample_rule).last()
        return active_lookback_df[self.name]
    
    def get_active_bouts_ts(self, resolution_u=30,
                            lookback_u=60,
                            min_active_u=1,
                            max_distance_u=1):
        """
        Count independent active bouts, based on minimum duration and minimum
        distance between activation parameters, within the last lookback_u 
        fullres_minutes units.
        Args:
            -resolution_u (int): resolution of the output dataframe, in 
                fullres_minutes units. Defaults to 30.
            -lookback_u (int): how amny fullres_minutes units to look back for 
                discovering activity bouts. Defaults to 60.
            -min_active_u (int): minimum active time, in fullres_minutes units.
                Defaults to 1.
            -max_distance_u (int): maximum distance between activations (in 
                fullres_minutes units), to be considered as part of the same 
                bout. Defaults to 1.
        Returns:
            -active_bouts_ts (pandas.TimeSeries): a time series with bouts 
                count within the last lookback_u fullres_minutes units
        """
        #Slow implementation! Needs to perform rolling computations
        steps = np.arange(start=lookback_u+resolution_u*(1+np.floor_divide(min_active_u,resolution_u)),
                          stop=self.fullres_m.shape[0]-1,
                          step=resolution_u)
        newindx = self.fullres_m.iloc[steps].index
        n_active_bouts_list = list()
        for t2 in steps:
            #test: t2=steps[3]
            t1 = t2-lookback_u-min_active_u+1
            xd = self.fullres_m.iloc[t1:t2].values.ravel()
            epoche = [(x,len(list(k))) for x,k in groupby(xd)]  #yields [(val,reps), (val,reps), ...]
            uni_outer = list()
            uni_inner=list()
            for b in epoche:
                if (b[0]==0 and b[1]>=max_distance_u):
                    uni_outer.append(uni_inner)
                    uni_inner=list()
                elif(b[0]==1):
                    uni_inner.append(b[1])
            if uni_inner:   #merge residual activations
                uni_outer.append(uni_inner)
            bouts_all = [np.sum(x) for x in uni_outer]
            n_bouts = np.sum([x>=min_active_u for x in bouts_all])
            n_active_bouts_list.append(n_bouts)
            
        active_bouts_ts = pd.Series(n_active_bouts_list,index=newindx,
                                    name='{}_BOUTS'.format(self.name))
        return active_bouts_ts
    
    def default_processing(self, sname, sdata, s_res, s_lookback,
                           s_min_active_u, s_max_distance_u):
        """
        Stramline processing.
        Args:
            -sname: sensor name
            -sdata: sensor data as timeseries
            -s_res: equivalent to resolution_u.
            -s_lookback: equivalent to lookback_u.
            -s_min_active_u: equivalent to min_active_u.
            -s_max_distance_u: equivalent to max_distance_u.
        Returns:
            -dictionary as follows:
                {'normalized_ts': normalized_ts from get_active_ts()
                'binary_ts': binary_ts from get_active_ts(),
                'active_bouts_ts': active_bouts_ts from get_active_bouts_ts(),
                'active_lookback_ts': active_lookback_ts from get_active_lookback_ts()}
        """
        self.compute_fullres_sigmatr(sdata)
        normalized_ts = self.get_active_ts(resolution_u=s_res,mode='normalized')
        binary_ts = self.get_active_ts(resolution_u=s_res, mode='binary',
                                        binary_minduration_u=s_min_active_u)
        active_bouts_ts = self.get_active_bouts_ts(resolution_u=s_res,
                                                    lookback_u=s_lookback,
                                                    min_active_u=s_min_active_u,
                                                    max_distance_u=s_max_distance_u)
        active_lookback_ts = self.get_active_lookback_ts(resolution_u=s_res,
                                            lookback_u=s_lookback)/s_lookback
        return {'normalized_ts':normalized_ts,
                'binary_ts':binary_ts,
                'active_bouts_ts':active_bouts_ts,
                'active_lookback_ts':active_lookback_ts}
        




class PirSensor(GENERICBINsensor):
    """
    Implements Bed Sensor methods, inheriting from GENERICBINsensor.
    """
    def __init__(self, name,
                 fullres_minutes=1,
                 timezone_adjust=None):
        """
        Deafults for pir are:
            -fullres_minutes=1
            -timezone_adjust=None
        Please look at the parent class for info on the parameters.
        """
        super().__init__(name=name,fullres_minutes=fullres_minutes,
                         timezone_adjust=timezone_adjust)
        return
    
    def compute_fullres_sigmatr(self,data,replace_minus1=True):
        """
        Create a signature matrix from an input dataframe
        Deafults for pir are:
            -replace_minus1=True
        Please look at the parent class for info on the parameters.
        """
        super().compute_fullres_sigmatr(data,replace_minus1=replace_minus1)
        return
    
    def get_active_ts(self, resolution_u=30,
                      mode='normalized',
                      binary_minduration_u=1):
        """
        Get an activation matrix, in a resampled resolution. Provides various
        output options: {'integral','normalized','binary'}. 
        Deafults for pir are:
            -resolution_u=30
            -mode='normalized'
            -binary_minduration_u=10 (if mode=='binary')
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_ts(resolution_u=resolution_u, mode=mode,
                                     binary_minduration_u=binary_minduration_u)
        
    def get_active_lookback_ts(self,resolution_u=30,
                               lookback_u=60):
        """
        Compute active time units within the last lookback_u, with an update
        resolution of resolution_u.
        Deafults for pir are:
            -resolution_u=30
            -lookback_u=60
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_lookback_ts(resolution_u=resolution_u,
                                              lookback_u=lookback_u)
        
    def get_active_bouts_ts(self, resolution_u=30,
                            lookback_u=60,
                            min_active_u=1,
                            max_distance_u=5):
        """
        Count independent active bouts, based on minimum duration and minimum
        distance between activation parameters, within the last lookback_u 
        fullres_minutes units.
        Deafults for pir are:
            -resolution_u=30
            -lookback_u=60
            -min_active_u=1
            -max_distance_u=5
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_bouts_ts(resolution_u=resolution_u,
                                           lookback_u=lookback_u,
                                           min_active_u=min_active_u,
                                           max_distance_u=max_distance_u)
        
    def default_processing(self,sname,sdata,s_res=30,s_lookback=60,
                           s_min_active_u=1,s_max_distance_u=5):
        """
        Stramline processing.
        Args:
            -sname: sensor name
            -sdata: sensor data as timeseries
            -s_res: equivalent to resolution_u.
            -s_lookback: equivalent to lookback_u.
            -s_min_active_u: equivalent to min_active_u.
            -s_max_distance_u: equivalent to max_distance_u.
        Returns:
            -dictionary as follows:
                {'normalized_ts': normalized_ts from get_active_ts()
                'binary_ts': binary_ts from get_active_ts(),
                'active_bouts_ts': active_bouts_ts from get_active_bouts_ts(),
                'active_lookback_ts': active_lookback_ts from get_active_lookback_ts()}
        """
        return super().default_processing(sname, sdata, s_res=s_res,
                        s_lookback=s_lookback, s_min_active_u=s_min_active_u,
                        s_max_distance_u=s_max_distance_u)




class ToiletSensor(GENERICBINsensor):
    """
    Implements Bed Sensor methods, inheriting from GENERICBINsensor.
    """
    def __init__(self, name,
                 fullres_minutes=1,
                 timezone_adjust=None):
        """
        Deafults for toilet are:
            -fullres_minutes=1
            -timezone_adjust=None
        Please look at the parent class for info on the parameters.
        """
        super().__init__(name=name,fullres_minutes=fullres_minutes,
                         timezone_adjust=timezone_adjust)
        return
    
    def compute_fullres_sigmatr(self,data,replace_minus1=True):
        """
        Create a signature matrix from an input dataframe
        Deafults for pir are:
            -replace_minus1=True
        Please look at the parent class for info on the parameters.
        """
        super().compute_fullres_sigmatr(data,replace_minus1=replace_minus1)
        return
    
    def get_active_ts(self, resolution_u=30,
                      mode='normalized',
                      binary_minduration_u=1):
        """
        Get an activation matrix, in a resampled resolution. Provides various
        output options: {'integral','normalized','binary'}. 
        Deafults for toilet are:
            -resolution_u=30
            -mode='normalized'
            -binary_minduration_u=10 (if mode=='binary')
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_ts(resolution_u=resolution_u, mode=mode,
                                     binary_minduration_u=binary_minduration_u)
        
    def get_active_lookback_ts(self,resolution_u=30,
                               lookback_u=60):
        """
        Compute active time units within the last lookback_u, with an update
        resolution of resolution_u.
        Deafults for toilet are:
            -resolution_u=30
            -lookback_u=60
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_lookback_ts(resolution_u=resolution_u,
                                              lookback_u=lookback_u)
        
    def get_active_bouts_ts(self, resolution_u=30,
                            lookback_u=60,
                            min_active_u=1,
                            max_distance_u=5):
        """
        Count independent active bouts, based on minimum duration and minimum
        distance between activation parameters, within the last lookback_u 
        fullres_minutes units.
        Deafults for toilet are:
            -resolution_u=30
            -lookback_u=60
            -min_active_u=1
            -max_distance_u=5
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_bouts_ts(resolution_u=resolution_u,
                                           lookback_u=lookback_u,
                                           min_active_u=min_active_u,
                                           max_distance_u=max_distance_u)
    
    def default_processing(self,sname,sdata,s_res=30,s_lookback=60,
                           s_min_active_u=1,s_max_distance_u=5):
        """
        Stramline processing.
        Args:
            -sname: sensor name
            -sdata: sensor data as timeseries
            -s_res: equivalent to resolution_u.
            -s_lookback: equivalent to lookback_u.
            -s_min_active_u: equivalent to min_active_u.
            -s_max_distance_u: equivalent to max_distance_u.
        Returns:
            -dictionary as follows:
                {'normalized_ts': normalized_ts from get_active_ts()
                'binary_ts': binary_ts from get_active_ts(),
                'active_bouts_ts': active_bouts_ts from get_active_bouts_ts(),
                'active_lookback_ts': active_lookback_ts from get_active_lookback_ts()}
        """
        return super().default_processing(sname, sdata, s_res=s_res,
                        s_lookback=s_lookback, s_min_active_u=s_min_active_u,
                        s_max_distance_u=s_max_distance_u)




class BedSensor(GENERICBINsensor):
    """
    Implements Bed Sensor methods, inheriting from GENERICBINsensor.
    """
    def __init__(self, name,
                 fullres_minutes=1,
                 timezone_adjust=None):
        """
        Deafults for bed are:
            -fullres_minutes=1
            -timezone_adjust=None
        Please look at the parent class for info on the parameters.
        """
        super().__init__(name=name,fullres_minutes=fullres_minutes,
                         timezone_adjust=timezone_adjust)
        return
    
    def compute_fullres_sigmatr(self,data,replace_minus1=True):
        """
        Create a signature matrix from an input dataframe
        Deafults for bed are:
            -replace_minus1=True
        Please look at the parent class for info on the parameters.
        """
        super().compute_fullres_sigmatr(data,replace_minus1=replace_minus1)
        return
    
    def get_active_ts(self, resolution_u=30,
                      mode='normalized',
                      binary_minduration_u=10):
        """
        Get an activation matrix, in a resampled resolution. Provides various
        output options: {'integral','normalized','binary'}. 
        Deafults for bed are:
            -resolution_u=30
            -mode='normalized'
            -binary_minduration_u=10 (if mode=='binary')
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_ts(resolution_u=resolution_u, mode=mode,
                                     binary_minduration_u=binary_minduration_u)
        
    def get_active_lookback_ts(self,resolution_u=30,
                               lookback_u=240):
        """
        Compute active time units within the last lookback_u, with an update
        resolution of resolution_u.
        Deafults for bed are:
            -resolution_u=30
            -lookback_u=240
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_lookback_ts(resolution_u=resolution_u,
                                              lookback_u=lookback_u)
        
    def get_active_bouts_ts(self, resolution_u=30,
                            lookback_u=60,
                            min_active_u=10,
                            max_distance_u=1):
        """
        Count independent active bouts, based on minimum duration and minimum
        distance between activation parameters, within the last lookback_u 
        fullres_minutes units.
        Deafults for bed are:
            -resolution_u=30
            -lookback_u=240
            -min_active_u=10
            -max_distance_u=1
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_bouts_ts(resolution_u=resolution_u,
                                           lookback_u=lookback_u,
                                           min_active_u=min_active_u,
                                           max_distance_u=max_distance_u)
        
    def default_processing(self,sname,sdata,s_res=30,s_lookback=60,
                           s_min_active_u=10,s_max_distance_u=1):
        """
        Stramline processing.
        Args:
            -sname: sensor name
            -sdata: sensor data as timeseries
            -s_res: equivalent to resolution_u.
            -s_lookback: equivalent to lookback_u.
            -s_min_active_u: equivalent to min_active_u.
            -s_max_distance_u: equivalent to max_distance_u.
        Returns:
            -dictionary as follows:
                {'normalized_ts': normalized_ts from get_active_ts()
                'binary_ts': binary_ts from get_active_ts(),
                'active_bouts_ts': active_bouts_ts from get_active_bouts_ts(),
                'active_lookback_ts': active_lookback_ts from get_active_lookback_ts()}
        """
        return super().default_processing(sname, sdata, s_res=s_res,
                        s_lookback=s_lookback, s_min_active_u=s_min_active_u,
                        s_max_distance_u=s_max_distance_u)
 



class ChairSensor(GENERICBINsensor):
    """
    Implements Chair Sensor methods, inheriting from GENERICBINsensor.
    """
    def __init__(self, name,
                 fullres_minutes=1,
                 timezone_adjust=None):
        """
        Deafults for chair are:
            -fullres_minutes=1
            -timezone_adjust=None
        Please look at the parent class for info on the parameters.
        """
        super().__init__(name=name,fullres_minutes=fullres_minutes,
                         timezone_adjust=timezone_adjust)
        return
    
    def compute_fullres_sigmatr(self,data,replace_minus1=True):
        """
        Create a signature matrix from an input dataframe
        Deafults for chair are:
            -replace_minus1=True
        Please look at the parent class for info on the parameters.
        """
        super().compute_fullres_sigmatr(data,replace_minus1=replace_minus1)
        return
    
    def get_active_ts(self, resolution_u=30,
                      mode='normalized',
                      binary_minduration_u=5):
        """
        Get an activation matrix, in a resampled resolution. Provides various
        output options: {'integral','normalized','binary'}. 
        Deafults for chair are:
            -resolution_u=30
            -mode='normalized'
            -binary_minduration_u=5 (if mode=='binary')
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_ts(resolution_u=resolution_u, mode=mode,
                                     binary_minduration_u=binary_minduration_u)
        
    def get_active_lookback_ts(self,resolution_u=30,
                               lookback_u=60):
        """
        Compute active time units within the last lookback_u, with an update
        resolution of resolution_u.
        Deafults for chair are:
            -resolution_u=30
            -lookback_u=60
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_lookback_ts(resolution_u=resolution_u,
                                              lookback_u=lookback_u)
        
    def get_active_bouts_ts(self, resolution_u=30,
                            lookback_u=60,
                            min_active_u=5,
                            max_distance_u=5):
        """
        Count independent active bouts, based on minimum duration and minimum
        distance between activation parameters, within the last lookback_u 
        fullres_minutes units.
        Deafults for chair are:
            -resolution_u=30
            -lookback_u=60
            -min_active_u=10
            -max_distance_u=1
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_bouts_ts(resolution_u=resolution_u,
                                           lookback_u=lookback_u,
                                           min_active_u=min_active_u,
                                           max_distance_u=max_distance_u)
    
    def default_processing(self,sname,sdata,s_res=30,s_lookback=60,
                           s_min_active_u=5,s_max_distance_u=5):
        """
        Stramline processing.
        Args:
            -sname: sensor name
            -sdata: sensor data as timeseries
            -s_res: equivalent to resolution_u.
            -s_lookback: equivalent to lookback_u.
            -s_min_active_u: equivalent to min_active_u.
            -s_max_distance_u: equivalent to max_distance_u.
        Returns:
            -dictionary as follows:
                {'normalized_ts': normalized_ts from get_active_ts()
                'binary_ts': binary_ts from get_active_ts(),
                'active_bouts_ts': active_bouts_ts from get_active_bouts_ts(),
                'active_lookback_ts': active_lookback_ts from get_active_lookback_ts()}
        """
        return super().default_processing(sname, sdata, s_res=s_res,
                        s_lookback=s_lookback, s_min_active_u=s_min_active_u,
                        s_max_distance_u=s_max_distance_u)





class ContactSensor(GENERICBINsensor):
    """
    Implements Chair Sensor methods, inheriting from GENERICBINsensor.
    """
    def __init__(self, name,
                 fullres_minutes=1,
                 timezone_adjust=None):
        """
        Deafults for contact are:
            -fullres_minutes=1
            -timezone_adjust=None
        Please look at the parent class for info on the parameters.
        """
        super().__init__(name=name,fullres_minutes=fullres_minutes,
                         timezone_adjust=timezone_adjust)
        return
    
    def compute_fullres_sigmatr(self,data,replace_minus1=True):
        """
        Create a signature matrix from an input dataframe
        Deafults for contact are:
            -replace_minus1=True
        Please look at the parent class for info on the parameters.
        """
        super().compute_fullres_sigmatr(data,replace_minus1=replace_minus1)
        return
    
    def get_active_ts(self, resolution_u=30,
                      mode='normalized',
                      binary_minduration_u=1):
        """
        Get an activation matrix, in a resampled resolution. Provides various
        output options: {'integral','normalized','binary'}. 
        Deafults for contact are:
            -resolution_u=30
            -mode='normalized'
            -binary_minduration_u=5 (if mode=='binary')
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_ts(resolution_u=resolution_u, mode=mode,
                                     binary_minduration_u=binary_minduration_u)
        
    def get_active_lookback_ts(self,resolution_u=30,
                               lookback_u=240):
        """
        Compute active time units within the last lookback_u, with an update
        resolution of resolution_u.
        Deafults for chair are:
            -resolution_u=30
            -lookback_u=60
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_lookback_ts(resolution_u=resolution_u,
                                              lookback_u=lookback_u)
        
    def get_active_bouts_ts(self, resolution_u=30,
                            lookback_u=60,
                            min_active_u=1,
                            max_distance_u=5):
        """
        Count independent active bouts, based on minimum duration and minimum
        distance between activation parameters, within the last lookback_u 
        fullres_minutes units.
        Deafults for contact are:
            -resolution_u=30
            -lookback_u=60
            -min_active_u=10
            -max_distance_u=1
        Please look at the parent class for info on the parameters.
        """
        return super().get_active_bouts_ts(resolution_u=resolution_u,
                                           lookback_u=lookback_u,
                                           min_active_u=min_active_u,
                                           max_distance_u=max_distance_u)
    
    def default_processing(self,sname,sdata,s_res=30,s_lookback=60,
                           s_min_active_u=1,s_max_distance_u=5):
        """
        Stramline processing.
        Args:
            -sname: sensor name
            -sdata: sensor data as timeseries
            -s_res: equivalent to resolution_u.
            -s_lookback: equivalent to lookback_u.
            -s_min_active_u: equivalent to min_active_u.
            -s_max_distance_u: equivalent to max_distance_u.
        Returns:
            -dictionary as follows:
                {'normalized_ts': normalized_ts from get_active_ts()
                'binary_ts': binary_ts from get_active_ts(),
                'active_bouts_ts': active_bouts_ts from get_active_bouts_ts(),
                'active_lookback_ts': active_lookback_ts from get_active_lookback_ts()}
        """
        return super().default_processing(sname, sdata, s_res=s_res,
                        s_lookback=s_lookback, s_min_active_u=s_min_active_u,
                        s_max_distance_u=s_max_distance_u)

    
    


if __name__ == '__main__':
    print('Tet Area')
    import matplotlib.pyplot as plt
    import seaborn as sns
    #if 'misure' not in locals():
    if True:
        #fetch data just the first time
        from activage_datautils import ActivageData
        ad = ActivageData()
        lista_users = ad.getRegisteredUsers()
        print('Utenti:\n{}'.format(lista_users))
        
        utente = lista_users[2]
        lista_sensori_1 = ad.getPersonSensorsList(utente)
        print('Lista sensori utente {}:\n{}'.format(utente,lista_sensori_1))
        
        misure = ad.getPersonMeasurements(personid=utente,
                                          sensorlist=lista_sensori_1,
                                          dateFrom='01-10-2018 00:01:01',
                                          dateTo='30-01-2019 23:59:59',
                                          measurementType='DATA')
        bed_id = [x for x in lista_sensori_1 if x.startswith('BE')][0]
        #chair_id = [x for x in lista_sensori_1 if x.startswith('CH')][0]
        toilet_id = [x for x in lista_sensori_1 if x.startswith('TO')][0]
        door_id = [x for x in lista_sensori_1 if x.startswith('CO')][0]
        pir_id = [x for x in lista_sensori_1 if x.startswith('PI')][0]
    
    #FIX AREA
    chair_id=bed_id
    
    
    #Bed
    s_bed = BedSensor(name='BED',timezone_adjust=None)
    bed_res = 30
    bed_lookback = 240
    bed_min_active_u = 10
    bed_max_distance_u = 1
    s_bed.compute_fullres_sigmatr(misure[bed_id])
    normalized_ts = s_bed.get_active_ts(resolution_u=bed_res,mode='normalized')
    binary_ts = s_bed.get_active_ts(resolution_u=bed_res, mode='binary',
                                    binary_minduration_u=bed_min_active_u)
    
    active_bouts_ts = s_bed.get_active_bouts_ts(resolution_u=bed_res,
                                                lookback_u=bed_lookback,
                                                min_active_u=bed_min_active_u,
                                                max_distance_u=bed_max_distance_u)
    tmp_mezzanotte = np.nonzero(active_bouts_ts.index.hour==0)[0][[0,-1]]
    nuovo_indice = active_bouts_ts.index[tmp_mezzanotte[0]:tmp_mezzanotte[1]-1]
    
    active_lookback_ts = s_bed.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback)/bed_lookback
    active_lookback_ts2 = s_bed.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback*2)/(bed_lookback*2)
    
    fig,ax= plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle('BED')
    ax[0][0].imshow(normalized_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][0].set_title('Instant')
    ax[0][1].imshow(active_lookback_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][1].set_title('Lookback 240')
    ax[1][1].imshow(active_lookback_ts2.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][1].set_title('Lookback 480')
    ax[1][0].imshow(active_bouts_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][0].set_title('Bouts 240')
    
    
    #Chair
    s_chair = ChairSensor(name='CHAIR',timezone_adjust=None)
    bed_res = 30
    bed_lookback = 120
    bed_min_active_u = 5
    bed_max_distance_u = 1
    s_chair.compute_fullres_sigmatr(misure[chair_id])
    normalized_ts = s_chair.get_active_ts(resolution_u=bed_res,mode='normalized')
    binary_ts = s_chair.get_active_ts(resolution_u=bed_res, mode='binary',
                                    binary_minduration_u=bed_min_active_u)
    
    active_bouts_ts = s_chair.get_active_bouts_ts(resolution_u=bed_res,
                                                lookback_u=bed_lookback,
                                                min_active_u=bed_min_active_u,
                                                max_distance_u=bed_max_distance_u)
    
    active_lookback_ts = s_chair.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback)/bed_lookback
    active_lookback_ts2 = s_chair.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback*2)/(bed_lookback*2)
    
    fig,ax= plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle('CHAIR')
    ax[0][0].imshow(normalized_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][0].set_title('Instant')
    ax[0][1].imshow(active_lookback_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][1].set_title('Lookback 120')
    ax[1][1].imshow(active_lookback_ts2.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][1].set_title('Lookback 240')
    ax[1][0].imshow(active_bouts_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][0].set_title('Bouts 120')
    
    #Toilet
    s_toilet = ToiletSensor(name='TOILET',timezone_adjust=None)
    bed_res = 30
    bed_lookback = 120
    bed_min_active_u = 1
    bed_max_distance_u = 5
    s_toilet.compute_fullres_sigmatr(misure[toilet_id])
    normalized_ts = s_toilet.get_active_ts(resolution_u=bed_res,mode='normalized')
    binary_ts = s_toilet.get_active_ts(resolution_u=bed_res, mode='binary',
                                    binary_minduration_u=bed_min_active_u)
    
    active_bouts_ts = s_toilet.get_active_bouts_ts(resolution_u=bed_res,
                                                lookback_u=bed_lookback,
                                                min_active_u=bed_min_active_u,
                                                max_distance_u=bed_max_distance_u)
    
    active_lookback_ts = s_toilet.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback)/bed_lookback
    active_lookback_ts2 = s_toilet.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback*2)/(bed_lookback*2)
    
    fig,ax= plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle('TOILET')
    ax[0][0].imshow(normalized_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][0].set_title('Instant')
    ax[0][1].imshow(active_lookback_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][1].set_title('Lookback 120')
    ax[1][1].imshow(active_lookback_ts2.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][1].set_title('Lookback 240')
    ax[1][0].imshow(active_bouts_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][0].set_title('Bouts 120')
    
    #Pir
    s_pir = PirSensor(name='PIR',timezone_adjust=None)
    bed_res = 30
    bed_lookback = 120
    bed_min_active_u = 1
    bed_max_distance_u = 5
    s_pir.compute_fullres_sigmatr(misure[pir_id])
    normalized_ts = s_pir.get_active_ts(resolution_u=bed_res,mode='normalized')
    binary_ts = s_pir.get_active_ts(resolution_u=bed_res, mode='binary',
                                    binary_minduration_u=bed_min_active_u)
    
    active_bouts_ts = s_pir.get_active_bouts_ts(resolution_u=bed_res,
                                                lookback_u=bed_lookback,
                                                min_active_u=bed_min_active_u,
                                                max_distance_u=bed_max_distance_u)
    
    active_lookback_ts = s_pir.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback)/bed_lookback
    active_lookback_ts2 = s_pir.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback*2)/(bed_lookback*2)
    
    fig,ax= plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle('PIR')
    ax[0][0].imshow(normalized_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][0].set_title('Instant')
    ax[0][1].imshow(active_lookback_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][1].set_title('Lookback 120')
    ax[1][1].imshow(active_lookback_ts2.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][1].set_title('Lookback 240')
    ax[1][0].imshow(active_bouts_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][0].set_title('Bouts 120')
    
    #Door
    s_door = ContactSensor(name='DOOR',timezone_adjust=None)
    bed_res = 30
    bed_lookback = 120
    bed_min_active_u = 1
    bed_max_distance_u = 5
    s_door.compute_fullres_sigmatr(misure[door_id])
    normalized_ts = s_door.get_active_ts(resolution_u=bed_res,mode='normalized')
    binary_ts = s_door.get_active_ts(resolution_u=bed_res, mode='binary',
                                    binary_minduration_u=bed_min_active_u)
    
    active_bouts_ts = s_door.get_active_bouts_ts(resolution_u=bed_res,
                                                lookback_u=bed_lookback,
                                                min_active_u=bed_min_active_u,
                                                max_distance_u=bed_max_distance_u)
    
    active_lookback_ts = s_door.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback)/bed_lookback
    active_lookback_ts2 = s_door.get_active_lookback_ts(resolution_u=bed_res,
                                        lookback_u=bed_lookback*2)/(bed_lookback*2)
    
    fig,ax= plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle('DOOR')
    ax[0][0].imshow(normalized_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][0].set_title('Instant')
    ax[0][1].imshow(active_lookback_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[0][1].set_title('Lookback 120')
    ax[1][1].imshow(active_lookback_ts2.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][1].set_title('Lookback 240')
    ax[1][0].imshow(active_bouts_ts.loc[nuovo_indice].values.reshape(-1,48),aspect='auto')
    ax[1][0].set_title('Bouts 120')
    
    
