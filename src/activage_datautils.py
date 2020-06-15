#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:43:28 2019

@author: Niccolò Mora
"""


import requests
from urllib.parse import urljoin
import json
import pandas as pd
import numpy as np
import logging
import sensors_classes



class ActivageData():
    def __init__(self,base_url=None,logger=None):
        """
        Parameters:
            -base_url (string): the URL to fetch information from and access sensor data
        """
        self.get_credentials(base_url)
        return
    
    def get_credentials(self,base_url=None):
        """
        Get credentials for accessing services. For now, provided by user
        Parameters:
            -base_url (string): the URL to fetch information from
        """
        self.base_url = base_url
        return
    
    def getRegisteredUsers(self):
        """
        Get users registered in the Activage DB
        Returns:
            -userslist (list): a list of all users
        """
        logging.info('Fetching registered users')
        nome_servizio = 'getRegisteredUsers'
        url_completo = urljoin(self.base_url,nome_servizio)
        r1 = requests.get(url_completo)
        userslist = json.loads(r1.text)['RESULTS']
        return userslist
    
    def getPersonSensorsList(self,personid):
        """
        Get users registered in the Activage DB
        Args:
            -personid (str): person ID of an Activage patient
        Returns:
            -sensorlist (list): a list of all sensors for the given personid
        """
        logging.info('Fetching sensor list for {}'.format(personid))
        nome_servizio = 'person/{}/sensor'.format(personid)
        url_completo = urljoin(self.base_url,nome_servizio)
        r1 = requests.get(url_completo)
        tmp = json.loads(r1.text)['RESULTS']
        sensorlist = [tmp[k]['id'] for k in range(len(tmp))]
        return sensorlist
    
    def getPersonMeasurements(self,personid,sensorlist,
                              dateFrom,dateTo,
                              measurementType=''):
        """
        Get all patient measurements
        Args:
            -personid (str): person ID of an Activage patient
            -sensorlist (list(str)): a list of all sensors to be queried
            -dateFrom (date, 'DD-MM-YYYY hh:mm:ss'): start date for sensor data
            -dateTo (date, 'DD-MM-YYYY hh:mm:ss'): start date for sensor data.
            -measurementType (str): measurementType parameter to be passed to 
                the API. Can be either:
                    *'DATA': retrieves only measurement (sensor status) data
                    *'BATTERY': retrieves only battery data
                    *'ALARM': retrieves only alarm data
                    *'': get all data
        Returns:
            -measurements_d (dict(pandas.TimeSeries or pandas.Dataframe)): 
                a dictionary of all requested measurements. 
                    * If measurementType={'DATA','BATTERY'}: each element of 
                        the dictionary is a pandas.TimeSeries
                    * Otherwise, elements are pandas DataFrame
        """
        infologger = ['Fetching measurements for {}'.format(personid),
                      '\t\t\t\tsensors = {}'.format(sensorlist),
                      '\t\t\t\tdateFrom = {}'.format(dateFrom),
                      '\t\t\t\tdateTo = {}'.format(dateTo),
                      '\t\t\t\tmeasurementType = {}'.format(measurementType)]
        logging.info('\n'.join(infologger))
        nome_servizio = 'measurements'
        url_completo = urljoin(self.base_url,nome_servizio)
        measurements_d = dict()
        formatodata = '%d-%m-%Y %H:%M:%S'
        
        for sensore in sensorlist:
            try:
                #logging.info('Fetching sensor {}'.format(sensore))
                reqdata={'sensor':{'id':sensore},
                         'dateFrom':dateFrom,
                         'dateTo':dateTo,
                         'patientId': personid,
                         'measurementType':measurementType}
                r1 = requests.post(url_completo,json.dumps(reqdata))
                misure = json.loads(r1.text)['RESULTS']
                misure = pd.DataFrame.from_dict(misure)
                misure['timestamp'] = pd.to_datetime(misure['timestamp'],
                                              format=formatodata)
                misure = misure.set_index('timestamp')
                misure = misure.sort_index()        # measurements are not always correctly sorted by timestamp!
                misure = misure[['value','measurementTypeDesc']]
                if measurementType in ['DATA','BATTERY']:
                    misure = misure[misure['measurementTypeDesc']==measurementType]
# =============================================================================
#                     misure = misure.astype(dtype={'measurementTypeDesc':str,
#                                                   'value':np.int})
# =============================================================================
                    misure = pd.to_numeric(misure['value'],errors='coerce')
                    misure = misure.fillna(method='ffill')
                    misure = misure.fillna(method='bfill')
                measurements_d.update({sensore:misure})
                logging.info('Fetched <{}> data'.format(sensore))
            except Exception as e:
                logging.error('\nException [{}]\n{}'.format(sensore,e), exc_info=True)
                measurements_d.update({sensore:pd.Series()})
                #raise e
        
        return measurements_d



def get_sensor_handle(sensorid,timezone_adjust=None):
    """
    Determine class of sensor and create an instance for handling
    """
    # Activage naming notation
    if sensorid[:2]=='PI':
        sh = sensors_classes.PirSensor(sensorid,timezone_adjust=timezone_adjust)
    elif sensorid[:2]=='TO':
        sh = sensors_classes.ToiletSensor(sensorid,timezone_adjust=timezone_adjust)
    elif sensorid[:2]=='BE':
        sh = sensors_classes.BedSensor(sensorid,timezone_adjust=timezone_adjust)
    elif sensorid[:2]=='CH':
        sh = sensors_classes.ChairSensor(sensorid,timezone_adjust=timezone_adjust)
    elif sensorid[:2]=='CO':
        sh = sensors_classes.ContactSensor(sensorid,timezone_adjust=timezone_adjust)
    else:
        raise ValueError('Sensor Class not understood: {}'.format(sensorid[:2]))
    return sh

    
    
    
    

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    import matplotlib.dates as mdates
    import matplotlib.cbook as cbook
    import datetime as dt
    import os
    import traceback
    
    logging.basicConfig(filename='../out/execlog.log',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.ERROR)
    print('Test Area')
    
    my_url = 'My/URL/To/Activage/Data'
    ad = ActivageData(base_url=my_url)
    lista_users = ad.getRegisteredUsers()
    print('Utenti:\n{}'.format(lista_users))
    
    # PARAMETRI
    date_low = dt.datetime(2019,9,1,0,0,1)
    date_hi = dt.datetime(2020,3,18,23,59,59)
#    date_hi = dt.datetime.now()
    interval_plt_week = 2
    
    formatoDataQuery = '%d-%m-%Y %H:%M:%S'
    date_low_str = date_low.strftime(formatoDataQuery)

    date_hi_str = date_hi.strftime(formatoDataQuery)
        
    figure_dir = '../out/figures/'
    dataout_dir = '../out/data/'
    
    for progressivo,utente in enumerate(lista_users):
        print('User {}/{} (ID: {})'.format(progressivo+1,len(lista_users),utente))
        lista_sensori_1 = ad.getPersonSensorsList(utente)
        print('Lista sensori user: {}\n'.format(lista_sensori_1))
        
        # Phase 1: get data
# =============================================================================
#         misure_utente = ad.getPersonMeasurements(personid=utente,
#                                                  sensorlist=lista_sensori_1,
#                                                  dateFrom='01-01-2019 00:00:01',
#                                                  dateTo='04-09-2019 23:59:59',
#                                                  measurementType='DATA')
#         misure_batteria = ad.getPersonMeasurements(personid=utente,
#                                                  sensorlist=lista_sensori_1,
#                                                  dateFrom='01-01-2019 00:00:01',
#                                                  dateTo='04-09-2019 23:59:59',
#                                                  measurementType='BATTERY')
# =============================================================================
        
        misure_utente = ad.getPersonMeasurements(personid=utente,
                                                 sensorlist=lista_sensori_1,
                                                 dateFrom=date_low_str,
                                                 dateTo=date_hi_str,
                                                 measurementType='DATA')
        misure_batteria = ad.getPersonMeasurements(personid=utente,
                                                 sensorlist=lista_sensori_1,
                                                 dateFrom=date_low_str,
                                                 dateTo=date_hi_str,
                                                 measurementType='BATTERY')
        
        # Phase 2: save data
        tmp_save_firstone = True
        for sname,sts in misure_utente.items():
            dataout_dest = os.path.join(dataout_dir,'{}.h5'.format(utente))
            #skip sensor if empty
            if sts.empty:
                continue
            if tmp_save_firstone:
                tmp_save_firstone = False
                sts.to_hdf(dataout_dest,key=sname,mode='w')
            else:
                sts.to_hdf(dataout_dest,key=sname,mode='a')
        
        # Phase 3: binary activation density
        analyses = dict()
        for sname,sts in misure_utente.items():
            try:
                sh = get_sensor_handle(sname)
                analyses[sname] = sh.default_processing(sname=sname,sdata=sts)
            except Exception:
                errmsg = '\n'.join(['\n\n*** User {}/{} (ID: {})'.format(progressivo+1,len(lista_users),utente),
                                'sname, sts = {}\n{}'.format(sname,sts),
                                'Error = \n{}'.format(traceback.format_exc())])
                print(errmsg)
                logging.error(errmsg)
        # Phase 4: plot data
        # Daily counts plot
        fig,ax = plt.subplots(len(lista_sensori_1),2,figsize=(12,12))
        fig.suptitle('User: {}\n'.format(utente))
        colori = ['blue','orange','crimson','purple','turquoise']
        try:
            for n,kv in enumerate(misure_utente.items()):
                k,v = kv
                tmp_cont = v.copy().to_frame()
                tmp_cont['giorno'] = v.index.date
                tmp_cont['Counts'] = tmp_cont['value']
                conteggi_giornalieri = tmp_cont.groupby('giorno').count()
                
                plt_years = mdates.YearLocator()   # every year
                plt_months = mdates.MonthLocator()  # every month
                plt_week = mdates.WeekdayLocator(interval=interval_plt_week)   #every interval_plt_week weeks
                plt_day = mdates.DayLocator()   #every day
                plt_fmt = mdates.DateFormatter('%Y-%m-%d')
                ax[n][0].xaxis.set_major_locator(plt_week)
                ax[n][0].xaxis.set_major_formatter(plt_fmt)
                ax[n][0].xaxis.set_minor_locator(plt_day)
                
                ax[n][0].scatter(conteggi_giornalieri.index.values,conteggi_giornalieri['value'],
                           label=k,alpha=0.3,c=colori[n])
                #ax[n][0].legend()
                ax[n][0].set_title('Sensor: {}'.format(k),color=colori[n])
                ax[n][0].set_xlim([date_low,date_hi])
                ax[n][0].set_ylim([0,conteggi_giornalieri['Counts'].max()*1.05])
                ax[n][0].format_xdata = mdates.DateFormatter('%Y-%m-%d')
                ax[n][0].tick_params(axis='x', rotation=30)
                sns.boxenplot(conteggi_giornalieri['Counts'],ax=ax[n][1],
                            color=colori[n],saturation=0.3)
                ax[n][1].set_xlim([0,conteggi_giornalieri['Counts'].max()*1.05])
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        except Exception:
            errmsg = '\n'.join(['\n\n*** User {}/{} (ID: {})'.format(progressivo+1,len(lista_users),utente),
                                'Daily counts plots',
                                'Shape({}) = {}'.format(k,v.shape),
                                'Error = \n{}'.format(traceback.format_exc())])
            print(errmsg)
            logging.error(errmsg)
        fig.savefig(os.path.join(figure_dir,'{}-counts.png'.format(utente)))
        plt.close(fig)
        
        # Battery plots
        try:
            fig2,ax2 = plt.subplots(figsize=(8,12))
            ax2.xaxis.set_major_locator(plt_week)
            ax2.xaxis.set_major_formatter(plt_fmt)
            ax2.xaxis.set_minor_locator(plt_day)
            ax2.set_title('User: {}\n'.format(utente))
            plt_years = mdates.YearLocator()   # every year
            plt_months = mdates.MonthLocator()  # every month
            plt_week = mdates.WeekdayLocator(interval=interval_plt_week)   #every interval_plt_week weeks
            plt_day = mdates.DayLocator()   #every day
            plt_fmt = mdates.DateFormatter('%Y-%m-%d')
            colori = ['blue','orange','crimson','purple','turquoise']
            for n,kv in enumerate(misure_batteria.items()):
                k,v = kv
                batteria = v.ewm(span=48,min_periods=1).mean()
                batteria = batteria/10
                ax2.step(batteria.index.values,batteria.values,
                           label=k,alpha=1,c=colori[n])
            ax2.hlines(4.4,date_low,date_hi,label='warning',linestyles ='dashed',
                       colors='grey',linewidth=3)
            ax2.hlines(3.7,date_low,date_hi,label='critical',linestyles ='dashed',
                       colors ='black',linewidth=3)
            ax2.set_xlim([date_low,date_hi])
            ax2.set_ylim([3.0,7.0])
            ax2.set_ylabel('Battery level [V]')
            ax2.set_xlabel('Date')
            ax2.legend()
            ax2.format_xdata = mdates.DateFormatter('%Y-%m-%d')
            ax2.tick_params(axis='x', rotation=30)
            fig2.tight_layout()
        except Exception:
            errmsg = '\n'.join(['\n\n*** User {}/{} (ID: {})'.format(progressivo+1,len(lista_users),utente),
                                'Battery plots',
                                'Shape({}) = {}'.format(k,v.shape),
                                'Error = \n{}'.format(traceback.format_exc())])
            print(errmsg)
            logging.error(errmsg)
        fig2.savefig(os.path.join(figure_dir,'{}-battery.png'.format(utente)))
        plt.close(fig2)
        
        # Profiles plot
        try:
            fig3,ax3= plt.subplots(len(lista_sensori_1),4,
                                   figsize=(15,25),
                                   sharex=True,sharey=True)
            fig3.suptitle('User: {}\n'.format(utente))
            acc_error_msg = ['Errori specifici plot:']
            for n,kv in enumerate(analyses.items()):
                try:
                    if n==0:
                        ax3[n][0].set_title('Normal. inst.')
                    nome = kv[0]
                    tmp_mezzanotte = np.nonzero(analyses[nome]['active_bouts_ts'].index.hour==0)[0][[0,-1]]
                    nuovo_indice = analyses[nome]['active_bouts_ts'].index[tmp_mezzanotte[0]:tmp_mezzanotte[1]-1]
                    ax3[n][0].set_ylabel(nome)
                    ax3[n][0].imshow(kv[1]['normalized_ts'].loc[nuovo_indice].values.reshape(-1,48),
                           cmap='Oranges',aspect='auto')
                except Exception:
                    acc_error_msg.append('{}:\n{}'.format('active_bouts_ts',
                                         traceback.format_exc()))
                try:
                    if n==0:
                        ax3[n][1].set_title('Bin. inst')
                    ax3[n][1].imshow(kv[1]['binary_ts'].loc[nuovo_indice].values.reshape(-1,48),
                       cmap='Oranges',aspect='auto')
                except Exception:
                    acc_error_msg.append('{}:\n{}'.format('active_bouts_ts',
                                         traceback.format_exc()))
                try:
                    if n==0:
                        ax3[n][2].set_title('Norm. last hr')
                    ax3[n][2].imshow(kv[1]['active_lookback_ts'].loc[nuovo_indice].values.reshape(-1,48),
                       cmap='Oranges',aspect='auto')
                except Exception:
                    acc_error_msg.append('{}:\n{}'.format('active_bouts_ts',
                                         traceback.format_exc()))
                try:
                    if n==0:
                        ax3[n][3].set_title('Bouts last hr')
                    ax3[n][3].imshow(kv[1]['active_bouts_ts'].loc[nuovo_indice].values.reshape(-1,48),
                       cmap='Oranges',aspect='auto')
                except Exception:
                    acc_error_msg.append('{}:\n{}'.format('active_bouts_ts',
                                         traceback.format_exc()))
            fig3.tight_layout(rect=[0, 0.06, 1, 0.94])
        except Exception:
            if len(acc_error_msg)==1:
                acc_error_msg.append('nessuno')
                acc_error_msg = ' '.join(acc_error_msg)
            else:
                acc_error_msg = '\n'.join(acc_error_msg)
            errmsg = '\n'.join(['\n\n*** User {}/{} (ID: {})'.format(progressivo+1,len(lista_users),utente),
                                'Profiles plots',
                                acc_error_msg,
                                'Shape({}) = {}'.format(nome,{ik:iv.shape for ik,iv in kv[1].items()}),
                                'Error = \n{}'.format(traceback.format_exc())])
            print(errmsg)
            logging.error(errmsg)
        fig3.savefig(os.path.join(figure_dir,'{}-data.png'.format(utente)),
                         dpi=96)
        plt.close(fig3)
    
