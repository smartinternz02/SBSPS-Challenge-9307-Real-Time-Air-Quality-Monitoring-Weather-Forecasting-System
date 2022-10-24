import requests
import json
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
import os
import tensorflow as tf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.base import prediction
from sklearn import metrics
from datetime import datetime, timedelta

def date_time():
    day = datetime.today()
    l=[]
    for i in range(84):
        end = day + timedelta(days=0,hours=2)
        start=end
        l.append(end)
    return l

def ad_test(dataset):
    dftest=adfuller(dataset,autolag='AIC')
    print("1.ADF",dftest[0])
    print("2.P-value",dftest[1]) 
    print("3.Number of Lags:",dftest[2])
    print("4.Num of Observations used for adf  regression and critical values critical calculation:",dftest[3])
    print("5.Critical values :")
    for key,val in dftest[4].items():
        print('\t',key,":" ,val)



def train():
    # df=df.dropna()
    ad_test(df['aqi'])
    stepwise_fit=auto_arima(df["aqi"],trace=True,supress_warnings=True)
    stepwise_fit.summary()
    train=df.iloc[:-30]
    test=df.iloc[-30:]
    model=ARIMA(df['aqi'], order=(2,0,2))
    model=model.fit()
    model.summary()
    pred=model.predict(start=1,end=83,type='levels')
    pred['Date']=pd.DataFrame(dates)

def aqipredict(latitude ,longitude):
    try: 
        url =urlapi() 
        querystring = {"lat":latitude,"lon":longitude,"hours":"16008"}
        headers = {
        "X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
        "X-RapidAPI-Host": "air-quality.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        data_past=response.text
        data_past=json.loads(data_past) 
        df_past= pd.DataFrame(data_past['data'])
        df_past['city']=data_past['city_name']
        df_past['latitude']=data_past['lat']
        df_past['longitude']=data_past['lon']
        df_past['timezone']=data_past['timezone']
        dates=date_time()
        df_1=pd.read_csv('files/datasets/aqi_predicted_hour_data.csv')
        df_1=pd.concat([df_past,df_1])
        df_1=df_1[[	'aqi','pm10','pm25','o3','timestamp_local','so2','no2','timestamp_utc','datetime','co','ts','city','latitude','longitude',	'timezone']]
        df_1.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
        os.remove("files/datasets/aqi_predicted_hour_data.csv")
        df_1.to_csv('files/datasets/aqi_predicted_hour_data.csv')
        df=pd.read_csv('files/datasets/aqi_predicted_hour_data.csv',index_col='datetime',parse_dates=True)
        #print(df.columns)
    except Exception as e:
        print(e)
        df_past=pd.read_csv('files\\datasets\\aqi_predicted_hour_data.csv')
        df_past=df_past.iloc[:99]
    return df_past


def weatherpredict(latitude,longitude):
    try:
        import requests
        url = "https://forecast9.p.rapidapi.com/rapidapi/forecast/"+str(round(float(latitude),5))+"/"+str(round(float(longitude),5))+"/summary/"
        headers = {
            "X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
            "X-RapidAPI-Host": "forecast9.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers)
        import json
        data=json.loads(response.text)
        data=data['items']
        import pandas as pd
        df=pd.json_normalize(data)
        return df
    except Exception as e:
        print("Error at weather api",e)

def urlapi():
    return "https://air-quality.p.rapidapi.com/forecast/airquality"



def newsupdates():
    # climate news
    try:
        import requests
        import json
        import pandas as pd
        url = "https://climate-change23.p.rapidapi.com/news"
        headers = {
            "X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
            "X-RapidAPI-Host": "climate-change23.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers)
        data=json.loads(response.text)
        df=pd.json_normalize(data)
        df_1=pd.read_csv('files/datasets/news.csv')
        df_1=pd.concat([df,df_1])
        df_1.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
        os.remove("files/datasets/news.csv")
        df_1.to_csv('files/datasets/news.csv')
        return df
    except:
        print()
        df_1=pd.read_csv('files/datasets/news.csv')
        return df_1