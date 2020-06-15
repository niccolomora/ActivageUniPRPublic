#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:50:54 2019

@author: Niccolò Mora
"""

import pandas as pd
import numpy as np
from src.activage_datautils import ActivageData
from datetime import datetime as dt
from datetime import timedelta
import xlsxwriter
import logging



class BatHistoryMissing(Exception):
    pass




class SensorHealthResult():
    def __init__(self,chrg_region='chrg_fail',
                last_est=0,
                last_seen=np.datetime64('0000-01-01 00:00:00'),
                elapsed_hrs_last=9999,
                msg_freq_hrs_50=9999,
                msg_freq_hrs_90=9999,
                msg_freq_hrs_99=9999):
        """
        Returns:
            -dictionary, with the following entries:
                + userid: initially None
                + sensorid': initially None
                + last_est: last estimated battery voltage
                + last_seen: last time the sensor sent a message
                + elapsed_hrs_last: time elapsed since last seen
                + chrg_region: charge status, according to the classification
                    in self._which_region()
                + msg_freq_hrs: 25,50,75,90,99 th percentiles of transmission 
                    intervals
        """
        self.result = {'userid':None,
                       'sensorid':None,
                       'chrg_region':chrg_region,
                       'last_est':np.round(last_est,decimals=2),
                       'last_seen':last_seen,
                       'elapsed_hrs_last':elapsed_hrs_last,
                       'msg_freq_hrs_50':msg_freq_hrs_50,
                       'msg_freq_hrs_90':msg_freq_hrs_90,
                       'msg_freq_hrs_99':msg_freq_hrs_99}
        return
    
    def add_userid(self,userid):
        self.result['userid'] = userid
        return
    
    def add_sensorid(self,sensorid):
        self.result['sensorid'] = sensorid
        return
    
    def to_dict(self):
        return self.result




class SensorHealthReport():
    def __init__(self,datefrom=None,days_back=30):
        """
        Args:
            -datefrom (string, optional, default=None): if defined, it is used 
                as start date for reporting. Must be in  'dd-mm-YYYY HH:MM:SS'
                format
            -days_back (integer, optional, default=30): if datefrom is not 
                defined, start date is current time-days_back days
        """
        self.ad = ActivageData()
        self.users = self.ad.getRegisteredUsers()
        self.user_kit = {u:self.ad.getPersonSensorsList(u) for u in self.users}
        self.formatodata = '%d-%m-%Y %H:%M:%S'
        self.dateto = dt.strftime(dt.now(),format=self.formatodata)
        
        try:
            if isinstance(datefrom,str):
                try:
                    self.datefrom = dt.strptime(datefrom,format=self.formatodata)
                except Exception:
                    raise ValueError('Incorrect datefrom parameter')
            elif datefrom is None:
                if isinstance(days_back,int):
                    tmp_dateform = dt.now()-timedelta(days=days_back)
                    tmp_dateform = dt(tmp_dateform.year,tmp_dateform.month,
                                       tmp_dateform.day,0,0,1)
                    self.datefrom = dt.strftime(tmp_dateform,format=self.formatodata)
                else:
                    raise ValueError('Incorrect days_back parameter')
        except Exception as e:
            logging.error('\n{}'.format(e), exc_info=True)
        return
    
    def make_report(self,outpath='../out/reportSensori.xlsx'):
        user_sensors_health_list = list()
        for u in self.users:
            print(u)
            misure = self.ad.getPersonMeasurements(personid=u,
                                           sensorlist=self.user_kit[u],
                                           dateFrom=self.datefrom,
                                           dateTo=self.dateto,
                                           measurementType='BATTERY')
            for s in self.user_kit[u]:
                try:
                    ba = SensorAnalyzer()
                    result = ba.analyze(misure[s],smoother='EWMA')
                except Exception as e:
                    print(e)
                    logging.error('\n{}'.format(e), exc_info=True)
                    result = SensorHealthResult()
                result.add_sensorid(sensorid=s)
                result.add_userid(userid=u)
                user_sensors_health_list.append(result.to_dict())
            user_sensors_health = pd.DataFrame(user_sensors_health_list)
            self._to_xlsx(df=user_sensors_health,outpath=outpath)
        return user_sensors_health
    
    def _to_xlsx(self,df,outpath):
        """
        Args:
            -df (pandas.DataFrame): the dataframe to convert to excel
            -outpath (string): name of the output file, with path
        """
        xls_map = dict()
        xls_map['A1'] = ('userid','Pilot ID')
        xls_map['B1'] = ('sensorid','Sensor ID')
        xls_map['C1'] = ('chrg_region','Charge Status')
        xls_map['D1'] = ('elapsed_hrs_last','Time Since Last Comm')
        xls_map['E1'] = ('last_est','Last Battery Level')
        xls_map['F1'] = ('last_seen','Last Comm Timestamp')
        xls_map['G1'] = ('msg_freq_hrs_50','Comm Freq [h] - median')
        xls_map['H1'] = ('msg_freq_hrs_90','Comm Freq [h] - 90th perc')
        xls_map['I1'] = ('msg_freq_hrs_99','Comm Freq [h] - 99th perc')
        ordered_columns = [xls_map['{}1'.format(l)][0] for l in 'ABCDEFGHI']
        
        with xlsxwriter.Workbook(outpath) as workbook:
            worksheet = workbook.add_worksheet('Report')
            
            #conditional formatting
            format_black = workbook.add_format({'bg_color': '#242424',
                               'font_color': '#FFFFFF'})
            format_red = workbook.add_format({'bg_color': '#ED4157',
                               'font_color': '#242424'})
            format_orangered = workbook.add_format({'bg_color': '#EB9C28',
                               'font_color': '#242424'})
            format_yellow = workbook.add_format({'bg_color': '#faef40',
                               'font_color': '#242424'})
            format_green = workbook.add_format({'bg_color': '#CCFF99',
                               'font_color': '#242424'})
            
            for kcol,knam in xls_map.items():
                worksheet.write(kcol, knam[1])
            # Start from the first cell below the headers.
            row = 1
            for _, dati in df.iterrows():
                col = 0
                # Convert the date string into a datetime object.
                for fillcol in ordered_columns:
                    if fillcol=='last_seen':
                        worksheet.write_string(row,col,
                                             dt.strftime(pd.to_datetime(dati['last_seen']),
                                                         '%d-%m-%Y %H:%M:%S'))
                    else:
                        worksheet.write(row, col,dati[fillcol])
                    col = col+1
                row = row+1
            #colorazione colonna chrg_region
            worksheet.conditional_format('C2:C{}'.format(row), {'type':     'text',
                                        'criteria': 'containing',
                                        'value':    'chrg_fail',
                                        'format':   format_black})
            worksheet.conditional_format('C2:C{}'.format(row), {'type':     'text',
                                        'criteria': 'containing',
                                        'value':    'chrg_critical',
                                        'format':   format_red})
            worksheet.conditional_format('C2:C{}'.format(row), {'type':     'text',
                                        'criteria': 'containing',
                                        'value':    'chrg_warn',
                                        'format':   format_orangered})
            worksheet.conditional_format('C2:C{}'.format(row), {'type':     'text',
                                        'criteria': 'containing',
                                        'value':    'chrg_good',
                                        'format':   format_green})
            #colorazione colonna elapsed_hrs_last
            worksheet.conditional_format('D2:D{}'.format(row), {'type':     'cell',
                                        'criteria': '>',
                                        'value':    120,
                                        'format':   format_black})
            worksheet.conditional_format('D2:D{}'.format(row), {'type':     'cell',
                                        'criteria': 'between',
                                        'minimum': 24,
                                        'maximum': 120,
                                        'format':   format_red})
            worksheet.conditional_format('D2:D{}'.format(row), {'type':     'cell',
                                        'criteria': 'between',
                                        'minimum': 12,
                                        'maximum': 24,
                                        'format':   format_orangered})
            worksheet.conditional_format('D2:D{}'.format(row), {'type':     'cell',
                                        'criteria': 'between',
                                        'minimum': 2,
                                        'maximum': 12,
                                        'format':   format_yellow})
            worksheet.conditional_format('D2:D{}'.format(row), {'type':     'cell',
                                        'criteria': 'between',
                                        'minimum': 0,
                                        'maximum': 2,
                                        'format':   format_green})
            
            
            
        return


class SensorAnalyzer():
    def __init__(self, chrg_good_inf=4.4,chrg_warn_inf=3.7):
        """
        Args:
            -chrg_good_inf (float, optional, default=4.4): lower bound for 
                chrg_good region
            -chrg_warn_inf (float, optional, default=3.7): lower bound for 
                chrg_warn region
        """
        self.chrg_good_inf = chrg_good_inf
        self.chrg_warn_inf = chrg_warn_inf
        self.bat_ts_smooth = None
        return
    
    
    def _which_region(self,hyst_days=30,hyst_increment=0.5):
        """
        Determine region, with hysteresis
        Args:
            -hyst_days (integer, optional, default=30): days back in time for
                the hysteresis
            -hyst_increment(float, optional, default=0.5): increment to each
                region threshold, for hysteresis
        Returns:
            - one of the following possible strings:
                {chrg_good,chrg_warn,chrg_critical}
        """
        if self.bat_ts_smooth is None:
            raise BatHistoryMissing()
        last_datetime = pd.to_datetime(self.bat_ts_smooth.index.values[-1])
        last_estim = self.bat_ts_smooth.iloc[-1]
        from_date = last_datetime-timedelta(days=hyst_days)
        hyst_obs = self.bat_ts_smooth.loc[from_date:]
        #Decide if chrg_critical
        if np.count_nonzero(hyst_obs.values<self.chrg_warn_inf):
            if last_estim<self.chrg_warn_inf+hyst_increment:
                return 'chrg_critical'
        #Decide if chrg_warn, since chrg_critical was discarded
        elif np.count_nonzero(hyst_obs.values<self.chrg_good_inf):
            if last_estim<self.chrg_good_inf+hyst_increment:
                return 'chrg_warn'
        else:
            return 'chrg_good'
    
    def analyze(self,bat_ts,smoother='EWMA',span=48,transmission_intv_nominal=60):
        """
        Args:
            -bat_ts (pandas.Series): a timeseries of battery level
            -smoother (string, optional, default='MA'): type of smoothing to
                apply: {'MA': moving Average, 'EWMA': Expon. Weighted MA}
            -span (integer, optional, default=24): span of the smoothing filter.
                In typical applications, 1 battery report is sent per hour
            -transmission_intv_nominal (integer, optional, default=60): 
                nominal inter-transmission interval, in minutes
        Returns
            -SensorHealthResult object
        """
        if not isinstance(bat_ts,pd.Series):
            raise ValueError('Invalid bat_ts argument (must be a pd.Series), got {}'.format(type(bat_ts)))
        elif bat_ts.empty:
            raise BatHistoryMissing()
        supported_smoothers = {'MA','EWMA'}
        if smoother not in supported_smoothers:
            raise ValueError('Invalid smoother. Supported types: {}'.format(supported_smoothers))
        if not isinstance(span,int):
            raise ValueError('Arg. span must be an integer')
        if smoother=='MA':
            self.bat_ts_smooth = bat_ts.rolling(span,min_periods=1).mean()
        elif smoother=='EWMA':
            self.bat_ts_smooth = bat_ts.ewm(span=span,min_periods=1).mean()
        self.last_est = self.bat_ts_smooth.iloc[-1]
        self.chrg_region = self._which_region(self.last_est)
        self.last_seen = pd.to_datetime(self.bat_ts_smooth.index.values[-1])
        self.elapsed_hrs_last = (dt.now()-pd.to_datetime(self.last_seen))/np.timedelta64(60,'m')
        #analyze transmission regularity
        transmissions = self.bat_ts_smooth.index.values
        transm_intv = np.diff(transmissions)/np.timedelta64(transmission_intv_nominal,'m')
        msg_freq_hrs_50 = np.round(np.percentile(transm_intv,50),3)
        msg_freq_hrs_90 = np.round(np.percentile(transm_intv,90),3)
        msg_freq_hrs_99 = np.round(np.percentile(transm_intv,99),3)
        result = SensorHealthResult(chrg_region=self.chrg_region,
                                    last_est=np.round(self.last_est,2),
                                    last_seen=self.last_seen,
                                    elapsed_hrs_last=np.round(self.elapsed_hrs_last,3),
                                    msg_freq_hrs_50=msg_freq_hrs_50,
                                    msg_freq_hrs_90=msg_freq_hrs_90,
                                    msg_freq_hrs_99=msg_freq_hrs_99)
        return result
        
        
    


if __name__ == '__main__':
    print('Test Area')
    
    
    logging.basicConfig(filename='../out/execlog.log',
                        format='\n%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.ERROR)
    
    br = SensorHealthReport(days_back=400)
    df = br.make_report()
