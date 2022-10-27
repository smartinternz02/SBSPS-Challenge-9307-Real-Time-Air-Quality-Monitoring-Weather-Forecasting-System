import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime 
from datetime import timedelta
import datetime
from datetime import time
from datetime import date
import requests
import json
import ibm_db
import calendar
import time
# importing graphs
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from mailjet_rest import Client
import os
# training models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix 



# importing modules
from modules.weather_apis import *
from modules.aqi_index_calculation import *
from modules.weather_prediction import *
from modules.aqi_api import *
#  from tempfile.timeseries import *


# Flask app
import flask
from flask import Flask, render_template,redirect, url_for, session,request



# ibm Database
conn = ibm_db.connect("DATABASE=bludb;HOSTNAME=21fecfd8-47b7-4937-840d-d791d0218660.bs2io90l08kqb1od8lcg.databases.appdomain.cloud;PORT=31864;SECURITY=SSL;SSLServerCertificate=DigiCertGlobalRootCA.crt;UID=vqf48690;PWD=FK4HnWZ5wLcaKVlc",'','')

location1=''

def Air_quality():
  dataset={'city_day':'https://drive.google.com/file/d/158j8UBocM-wzIF29fsiBVAmfwQA2JVIV/view?usp=sharing',
          'city_hour' :'https://drive.google.com/file/d/1vNRx81y6CehUR81t9oNiyirrE3F7Rwzj/view?usp=sharing',
          'station_day':'https://drive.google.com/file/d/1M6oxdKEjNflQh4euBTsFCslmXOjBi1wq/view?usp=sharing',
          'station_hour':'https://drive.google.com/file/d/1QyQY6vv4Ul_wsw69ZBsZrn69uD7Q8Vti/view?usp=sharing',
          'stations':'https://drive.google.com/file/d/1gP49xt_l-B2R3LNKBbPdCmje8fYmzr_T/view?usp=sharing'}

  ## defining dataset paths

  PATH_STATION_DAY = 'https://drive.google.com/uc?id=' + dataset['station_day'].split('/')[-2]
  PATH_STATION_HOUR = 'https://drive.google.com/uc?export=download&confirm=CONFIRM_CODE&id=1QyQY6vv4Ul_wsw69ZBsZrn69uD7Q8Vti'
  PATH_CITY_HOUR = 'https://drive.google.com/uc?id=' + dataset['city_hour'].split('/')[-2]
  PATH_CITY_DAY = 'https://drive.google.com/uc?id=' + dataset['city_day'].split('/')[-2]
  PATH_STATIONS = 'https://drive.google.com/uc?id=' + dataset['stations'].split('/')[-2]
  STATIONS = ["KL007", "KL008"]

  ## importing data and subsetting the station
  df = pd.read_csv(PATH_STATION_HOUR, parse_dates = ["Datetime"])
  stations = pd.read_csv(PATH_STATIONS)

  df = df.merge(stations, on = "StationId")

  df = df[df.StationId.isin(STATIONS)]
  df.sort_values(["StationId", "Datetime"], inplace = True)
  df["Date"] = df.Datetime.dt.date.astype(str)
  df.Datetime = df.Datetime.astype(str)
  df.fillna(0,inplace=True)

  df["PM10_24hr_avg"] = df.groupby("StationId")["PM10"].rolling(window = 24, min_periods = 16).mean().values
  df["PM2.5_24hr_avg"] = df.groupby("StationId")["PM2.5"].rolling(window = 24, min_periods = 16).mean().values
  df["SO2_24hr_avg"] = df.groupby("StationId")["SO2"].rolling(window = 24, min_periods = 16).mean().values
  df["NOx_24hr_avg"] = df.groupby("StationId")["NOx"].rolling(window = 24, min_periods = 16).mean().values
  df["NH3_24hr_avg"] = df.groupby("StationId")["NH3"].rolling(window = 24, min_periods = 16).mean().values
  df["CO_8hr_max"] = df.groupby("StationId")["CO"].rolling(window = 8, min_periods = 1).max().values
  df["O3_8hr_max"] = df.groupby("StationId")["O3"].rolling(window = 8, min_periods = 1).max().values


  df["SO2_SubIndex"] = df["SO2_24hr_avg"].apply(lambda x: get_SO2_subindex(x))
  df["NOx_SubIndex"] = df["NOx_24hr_avg"].apply(lambda x: get_NOx_subindex(x))
  df["O3_SubIndex"] = df["O3_8hr_max"].apply(lambda x: get_O3_subindex(x))
  df["CO_SubIndex"] = df["CO_8hr_max"].apply(lambda x: get_CO_subindex(x))
  df["PM10_SubIndex"] = df["PM10_24hr_avg"].apply(lambda x: get_PM10_subindex(x))
  df["PM2.5_SubIndex"] = df["PM2.5_24hr_avg"].apply(lambda x: get_PM25_subindex(x))
  df["NH3_SubIndex"] = df["NH3_24hr_avg"].apply(lambda x: get_NH3_subindex(x))

  df["Checks"] = (df["PM2.5_SubIndex"] > 0).astype(int) + \
                  (df["PM10_SubIndex"] > 0).astype(int) + \
                  (df["SO2_SubIndex"] > 0).astype(int) + \
                  (df["NOx_SubIndex"] > 0).astype(int) + \
                  (df["NH3_SubIndex"] > 0).astype(int) + \
                  (df["CO_SubIndex"] > 0).astype(int) + \
                  (df["O3_SubIndex"] > 0).astype(int)

  df["AQI_calculated"] = round(df[["PM2.5_SubIndex", "PM10_SubIndex", "SO2_SubIndex", "NOx_SubIndex",
                                  "NH3_SubIndex", "CO_SubIndex", "O3_SubIndex"]].max(axis = 1))
  df.loc[df["PM2.5_SubIndex"] + df["PM10_SubIndex"] <= 0, "AQI_calculated"] = np.NaN
  df.loc[df.Checks < 3, "AQI_calculated"] = np.NaN

  df["AQI_bucket_calculated"] = df["AQI_calculated"].apply(lambda x: get_AQI_bucket(x))
  df[~df.AQI_calculated.isna()].head(13)


  df_station_hour = df
  df_station_day = pd.read_csv(PATH_STATION_DAY)

  df_station_day = df_station_day.merge(df.groupby(["StationId", "Date"])["AQI_calculated"].mean().reset_index(), on = ["StationId", "Date"])
  df_station_day.AQI_calculated = round(df_station_day.AQI_calculated)


  df_city_hour = pd.read_csv(PATH_CITY_HOUR)
  df_city_day = pd.read_csv(PATH_CITY_DAY)

  df_city_hour["Date"] = pd.to_datetime(df_city_hour.Datetime).dt.date.astype(str)

  df_city_hour = df_city_hour.merge(df.groupby(["City", "Datetime"])["AQI_calculated"].mean().reset_index(), on = ["City", "Datetime"])
  df_city_hour.AQI_calculated = round(df_city_hour.AQI_calculated)

  df_city_day = df_city_day.merge(df_city_hour.groupby(["City", "Date"])["AQI_calculated"].mean().reset_index(), on = ["City", "Date"])
  df_city_day.AQI_calculated = round(df_city_day.AQI_calculated)


  df_check_station_hour = df_station_hour[["AQI", "AQI_calculated"]].dropna()
  df_check_station_day = df_station_day[["AQI", "AQI_calculated"]].dropna()
  df_check_city_hour = df_city_hour[["AQI", "AQI_calculated"]].dropna()
  df_check_city_day = df_city_day[["AQI", "AQI_calculated"]].dropna()

  df1=pd.read_csv(PATH_STATIONS)
  df=pd.merge(df1,df,on='StationId')  

  df = df.dropna()  

  df=pd.read_csv('files/datasets/aqi_data.csv')
  X=df[['PM2.5_SubIndex','PM10_SubIndex','SO2_SubIndex', 'NOx_SubIndex', 'NH3_SubIndex', 'CO_SubIndex','O3_SubIndex',]]
  Y=df[['AQI_calculated']]
  X.tail(10)

  X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=70)
  print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

  RF=RandomForestRegressor().fit(X_train,Y_train)
  train_preds1=RF.predict(X_train)
  test_preds1=RF.predict(X_test)
  RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train, train_preds1)))
  RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds1)))
  print("RMSE TrainingData ", str(RMSE_train))
  print("RMSE TestData", str(RMSE_test))
  print('-'*50)
  print('RSquared value on train:',RF.score (X_train, Y_train))
  print('RSquared value on test:',RF.score (X_test, Y_test))
  url = "https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"

  querystring = {"city":"new delhi"}

  headers = {
    "X-RapidAPI-Key": "4f7bb14128msh9b291ceae57c2d4p12b8b5jsn9580099f1b46",
    "X-RapidAPI-Host": "air-quality-by-api-ninjas.p.rapidapi.com"
  }

  response = requests.request("GET", url, headers=headers, params=querystring)

  print(response.text)
  data=response.text
  data=json.loads(data)
  print(data["CO"]['concentration'])


def weatherdays(test):
  j=0
  test['Day']=''
  for i in range(len(test['temp'])):
    if(j>7):
      break
    date=datetime.today()+timedelta(days=j)
    month=date.strftime('%b')
    da=date.strftime('%d')
    test['Day'][i]=month+" "+da
    if(test['temp'][i]<=24 and test['temp'][i]<=36):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=22 and test['temp'][i]<=34):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=21 and test['temp'][i]<=30):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=16 and test['temp'][i]<=32):
      print(date," ",test['temp'][i],test['weather'][i])
    elif(test['temp'][i]<=16 and test['temp'][i]<=32):
      print(date," ",test['temp'][i],test['weather'][i])
    j+=1
  return test


app = Flask(__name__)



@app.route('/',methods=['GET','POST'])    
def login():
    if flask.request.method == 'POST': 
      try:
        location=request.form.get('location')
        latitude=request.form.get('latitude')
        longitude=request.form.get('longitude')
        print(location,longitude,latitude)
        df=weather_pastdata(latitude,longitude)
        print("home / ",df)
        weather=weatherdays(df)
        df=df[['timestamp_local','temp']]
        df=df.rename(columns={'timestamp_local':'Timeline','temp':'Temperature'})
        fig = px.line(df, x="Timeline", y="Temperature",title="Weather Forecasting (Celsius)"+location)
        graph_weather = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
        fig1 = px.line(weather, x="Day", y="temp",title="Weather Forecasting (Celsius) of "+location)
        graph_weather1 = json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)

        # ==========================================================

        # ==========================================================
        return render_template('index.html',weather=weather.iloc[1:7],graph_weather=graph_weather,today=weather.iloc[0],graph_weather1=graph_weather1,location=location)
      except Exception as e:
        print("Error---------------- / ",e)
        df=pd.read_csv('files/datasets/weather.csv')
        weather=weatherdays(df)
        df=df[['timestamp_local','temp']]
        df=df.rename(columns={'timestamp_local':'Timeline','temp':'Temperature'})
        fig = px.line(df, x="Timeline", y="Temperature",title="Weather Forecasting (Celsius) of "+location)
        graph_weather = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
        fig1 = px.line(weather, x="Day", y="temp",title="Weather Forecasting (Celsius) of "+location)
        graph_weather1 = json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('index.html',weather=weather.iloc[1:7],graph_weather=graph_weather,today=weather.iloc[0],graph_weather1=graph_weather1,location=location)
    else:
      try:
        df=weather_pastdata(28.7041,77.1025)
        weather=weatherdays(df)
        df=df[['timestamp_local','temp']]
        df=df.rename(columns={'timestamp_local':'Timeline','temp':'Temperature'})
        fig = px.line(df, x="Timeline", y="Temperature",title="Weather Forecasting (Celsius) of New Delhi")
        graph_weather = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
        fig1 = px.line(weather, x="Day", y="temp",title="Weather Forecasting (Celsius) of New Delhi")
        graph_weather1 = json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('index.html',weather=weather.iloc[1:7],graph_weather=graph_weather,today=weather.iloc[0],graph_weather1=graph_weather1,location="New Delhi")
      except :
        print("error home get")
        df=pd.read_csv('files/datasets/weather.csv')
        weather=weatherdays(df)
        df=df[['timestamp_local','temp']]
        df=df.rename(columns={'timestamp_local':'Timeline','temp':'Temperature'})
        fig = px.line(df, x="Timeline", y="Temperature",title="Weather Forecasting (Celsius) of New Delhi")
        graph_weather = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
        fig1 = px.line(weather, x="Day", y="temp",title="Weather Forecasting (Celsius) of New Delhi ")
        graph_weather1 = json.dumps(fig1,cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('index.html',weather=weather.iloc[1:7],graph_weather=graph_weather,today=weather.iloc[0],graph_weather1=graph_weather1,location="New Delhi")








@app.route('/aqi',methods=['GET','POST'])
def aqi():
        if flask.request.method == 'POST':  
            #print(request.args.get('location'))
            location=request.form.get('location')
            # location1=request.form.get('location')
            latitude=request.form.get('latitude')
            longitude=request.form.get('longitude')
            location=location.split(',')
            location=location[0]
            if(len(latitude)>0 and len(longitude)>0):
                #data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109}
                date = datetime.today()
                month=date.strftime('%b')
                da=date.strftime('%d')
                df=aqipredict(latitude,longitude)
                df=df.rename(columns={'aqi':'AQI','so2':'SO2','no2':'NO2','pm10':'PM10','pm25':'PM2.5','co':'CO','o3':'O3','timestamp_local':'Date-Time'})
                fig_aqi= px.bar(df, x="Date-Time", y='AQI',color="AQI",  barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="AQI of "+location)
                fig_so2 = px.bar(df, x="Date-Time", y='SO2', color="SO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="SO2 Concentration of "+location)
                fig_no2= px.bar(df, x="Date-Time", y='NO2', color="NO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="NO2 Concentrations of "+location)
                fig_o3 = px.bar(df, x="Date-Time", y='O3', color="O3", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="O3 Concentrations of "+location)
                fig_co= px.bar(df, x="Date-Time", y='CO', color="CO", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="CO Concentrations of "+location)
                fig_PM10= px.bar(df, x="Date-Time", y='PM10', color="PM10", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM10 Concentrations of "+location)
                fig_PM25= px.bar(df, x="Date-Time", y='PM2.5', color="PM2.5", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM2.5 Concentrations of"+location)
                graph_aqi = json.dumps(fig_aqi,cls=plotly.utils.PlotlyJSONEncoder)
                graph_so2= json.dumps(fig_so2,cls=plotly.utils.PlotlyJSONEncoder)
                graph_no2= json.dumps(fig_no2,cls=plotly.utils.PlotlyJSONEncoder)
                graph_o3= json.dumps(fig_o3,cls=plotly.utils.PlotlyJSONEncoder)
                graph_co= json.dumps(fig_co,cls=plotly.utils.PlotlyJSONEncoder)
                graph_pm10= json.dumps(fig_PM10,cls=plotly.utils.PlotlyJSONEncoder)
                graph_pm25= json.dumps(fig_PM25,cls=plotly.utils.PlotlyJSONEncoder)
                # return render_template('graph.html',graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)
                return render_template('aqi.html',data={'CO': {'max':df['CO'].max(), 'min':df['CO'].min(),'avg':df['CO'].mean() },
                                                         'NO2': {'max':df['NO2'].max(), 'min':df['NO2'].min(),'avg':df['NO2'].mean() }, 
                                                         'O3': {'max':df['O3'].max(), 'min':df['O3'].min(),'avg':df['O3'].mean() },
                                                         'SO2': {'max':df['SO2'].max(), 'min':df['SO2'].min(),'avg':df['SO2'].mean() },
                                                         'PM2.5': {'max':df['PM2.5'].max(), 'min':df['PM2.5'].min(),'avg':df['PM2.5'].mean() },
                                                         'PM10': {'max':df['PM10'].max(), 'min':df['PM10'].min(),'avg':df['PM10'].mean() } },
                                                         AQI={'max':df['AQI'].max(),'avg':df['AQI'].mean(),'min':df['AQI'].min()},month=month,date=da,graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25,location=location)
            else:
                return render_template('404.html')
        elif flask.request.method == 'GET':
            loc="Delhi"
            date = datetime.today()
            month=date.strftime('%b')
            da=date.strftime('%d')
            df=aqipredict(28.7041,77.1025)
            df=df.rename(columns={'aqi':'AQI','so2':'SO2','no2':'NO2','pm10':'PM10','pm25':'PM2.5','co':'CO','o3':'O3','timestamp_local':'Date-Time'})
            fig_aqi= px.bar(df, x="Date-Time", y='AQI',color="AQI",  barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="AQI of "+loc)
            fig_so2 = px.bar(df, x="Date-Time", y='SO2', color="SO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="SO2 Concentration of "+loc)
            fig_no2= px.bar(df, x="Date-Time", y='NO2', color="NO2", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="NO2 Concentrations of "+loc)
            fig_o3 = px.bar(df, x="Date-Time", y='O3', color="O3", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="O3 Concentrations of "+loc)
            fig_co= px.bar(df, x="Date-Time", y='CO', color="CO", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="CO Concentrations of "+loc)
            fig_PM10= px.bar(df, x="Date-Time", y='PM10', color="PM10", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM10 Concentrations of "+loc)
            fig_PM25= px.bar(df, x="Date-Time", y='PM2.5', color="PM2.5", barmode="stack",color_continuous_scale=["green", "yellow","orange","red"],title="PM2.5 Concentrations of"+loc)
            graph_aqi = json.dumps(fig_aqi,cls=plotly.utils.PlotlyJSONEncoder)
            graph_so2= json.dumps(fig_so2,cls=plotly.utils.PlotlyJSONEncoder)
            graph_no2= json.dumps(fig_no2,cls=plotly.utils.PlotlyJSONEncoder)
            graph_o3= json.dumps(fig_o3,cls=plotly.utils.PlotlyJSONEncoder)
            graph_co= json.dumps(fig_co,cls=plotly.utils.PlotlyJSONEncoder)
            graph_pm10= json.dumps(fig_PM10,cls=plotly.utils.PlotlyJSONEncoder)
            graph_pm25= json.dumps(fig_PM25,cls=plotly.utils.PlotlyJSONEncoder)
            # return render_template('graph.html',graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25)
            return render_template('aqi.html',data={'CO': {'max':df['CO'].max(), 'min':df['CO'].min(),'avg':df['CO'].mean() },
                                                         'NO2': {'max':df['NO2'].max(), 'min':df['NO2'].min(),'avg':df['NO2'].mean() }, 
                                                         'O3': {'max':df['O3'].max(), 'min':df['O3'].min(),'avg':df['O3'].mean() },
                                                         'SO2': {'max':df['SO2'].max(), 'min':df['SO2'].min(),'avg':df['SO2'].mean() },
                                                         'PM2.5': {'max':df['PM2.5'].max(), 'min':df['PM2.5'].min(),'avg':df['PM2.5'].mean() },
                                                         'PM10': {'max':df['PM10'].max(), 'min':df['PM10'].min(),'avg':df['PM10'].mean() } },
                                                         AQI={'max':df['AQI'].max(),'avg':df['AQI'].mean(),'min':df['AQI'].min()},month=month,date=da,graph_aqi=graph_aqi,graph_so2=graph_so2,graph_no2=graph_no2,graph_o3=graph_o3,graph_co=graph_co,graph_pm10=graph_pm10,graph_pm25=graph_pm25,location="Delhi")
        else:
          return render_template('404.html')
       
@app.route('/find-aqi-of-place',methods=['POST'])
def find_aqi():
    #print(request.get_data)
    x = [request.form['autocomplete'],request.form['latitude'],request.form['longitude']]
    print(x)
    if(len(x[1])>0 and len(x[2])>0):
        location=x[0]
        latitude=x[1]
        longitude=x[2]
        #data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109}
        date = datetime.today()
        month=date.strftime('%b')
        da=date.strftime('%d')
        #print('-------------------========================================================-======',date,month,da
    
        return render_template('aqi.html',data={'CO': {'concentration': 961.3, 'aqi': 10}, 'NO2': {'concentration': 50.04, 'aqi': 62}, 'O3': {'concentration': 30.76, 'aqi': 26}, 'SO2': {'concentration': 79.16, 'aqi': 70}, 'PM2.5': {'concentration': 45.22, 'aqi': 109}, 'PM10': {'concentration': 57.56, 'aqi': 51}, 'overall_aqi': 109},month=month,date=da)
    return render_template('404.html')



@app.route('/news')    
def news():
    return render_template('news.html')

@app.route('/Subscribe')    
def contact():
    return render_template('contact.html')

@app.route('/live-cameras')    
def live_cameras():
    return render_template('404.html')

@app.route('/photos')    
def photos():
    return render_template('404.html')

@app.route('/404')
def notfound_404():
    return render_template('404.html')



  

@app.route('/subscribe', methods =['GET', 'POST'])
def registet():
    msg = ''
    if request.method == 'POST' :
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        location = request.form['location']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        sql = "SELECT * FROM users WHERE email =? OR mobile=?"
        stmt = ibm_db.prepare(conn, sql)
        ibm_db.bind_param(stmt,1,email)
        ibm_db.bind_param(stmt,2,mobile)
        ibm_db.execute(stmt)
        account = ibm_db.fetch_assoc(stmt)
        print(account)
        if account:
            msg = 'Account already exists !'
            return msg
        else:
            insert_sql = "INSERT INTO  users VALUES (?,?,?,?,?,?)"
            prep_stmt = ibm_db.prepare(conn, insert_sql)
            ibm_db.bind_param(prep_stmt,1,name)
            ibm_db.bind_param(prep_stmt,2,email)
            ibm_db.bind_param(prep_stmt,3,mobile)
            ibm_db.bind_param(prep_stmt,4,location)
            ibm_db.bind_param(prep_stmt,5,latitude)
            ibm_db.bind_param(prep_stmt,6,longitude) 
            ibm_db.execute(prep_stmt)   
            msg = 'You have successfully registered !'
            try :
                df=aqipredict(latitude,longitude)
                df1=weatherpredict(latitude,longitude)
                df=df.rename(columns={'aqi':'AQI','so2':'SO2','no2':'NO2','pm10':'PM10','pm25':'PM2.5','co':'CO','o3':'O3','timestamp_local':'Date-Time'})
                df=df[['Date-Time','AQI','PM10','PM2.5','SO2','NO2','CO','O3']]
                srt= " <h3>Hellow "+name+",</h3> <h3 style='color:green'> Welcome to <a href='http://127.0.0.1:5000/'>Real Time Weather Fore Casting</a>!<br/>  <h3 style='color :blue'> Location : "+location+"<br> Latitude : "+latitude+" Longitude : "+longitude+"</h3>"+"<h2 style='color: purple'>----------------------------  One week Report on Air Quality Index ----------------------------</h2> "+(df.describe()).to_html(classes='table table-stripped')+"<h2 style='color: purple'>----------------------------  One week updates on Air Quality Index ----------------------------</h2> "+df.to_html(classes='table table-stripped')+""
                from mailjet_rest import Client
                import os
                api_key = '45a5d104834c16dff39c75439e26a550'
                api_secret = '40b35dc277f9b9e82ac3eb342767cc15'
                mailjet = Client(auth=(api_key, api_secret), version='v3.1')
                data = {
                  'Messages': [
                    {
                      "From": {
                        "Email": "209x1a05h6@gprec.ac.in",
                        "Name": "Real Time Air Quality and Weather Forecasting"
                      },
                      "To": [
                        {
                          "Email": email,
                          "Name": name
                        }
                      ],
                      "Subject": "Today Updates from Real Time Air Quality and Weather Forecasting",
                      "TextPart": "Subscribe to get Latest Updates",
                      "HTMLPart": srt,
                      "CustomID": "AppGettingStartedTest"
                    }
                  ]
                }
                result = mailjet.send.create(data=data)
            except:
                print("Error at email sending")
            return render_template('success.html')




if __name__ == "__main__":
    app.run(debug=True)

















# @app.route('/login',methods =['GET', 'POST'])
# def login():
#     global userid
#     msg = ''
#     if request.method == 'POST' :
#         username = request.form['username']
#         password = request.form['password']
#         sql = "SELECT * FROM users WHERE username =? AND password=?"
#         stmt = ibm_db.prepare(conn, sql)
#         ibm_db.bind_param(stmt,1,username)
#         ibm_db.bind_param(stmt,2,password)
#         ibm_db.execute(stmt)
#         account = ibm_db.fetch_assoc(stmt)
#         print (account)
#         if account:
#             session['loggedin'] = True
#             session['id'] = account['USERNAME']
#             userid=  account['USERNAME']
#             session['username'] = account['USERNAME']
#             msg = 'Logged in successfully !'
            
#             msg = 'Logged in successfully !'
#         else:
#             msg = 'Incorrect username / password !'
#     return render_template('404.html', msg = msg)



# @app.route('/s', methods=['POST'])
# def subscribe():
#     msg = ''
#     if request.method == 'POST' :
#         name = request.form['name']
#         email = request.form['email']
#         mobile = request.form['mobile']
#         location = request.form['location']
#         latitude = request.form['latitude']
#         longitude = request.form['longitude']
#         print(latitude,longitude)
#         insert_sql = "INSERT INTO  users VALUES (?,?,?,?,?,?)"
#         prep_stmt = ibm_db.prepare(conn, insert_sql)
#         ibm_db.bind_param(prep_stmt,1,name)
#         ibm_db.bind_param(prep_stmt,2,email)
#         ibm_db.bind_param(prep_stmt,3,mobile)
#         ibm_db.bind_param(prep_stmt,4,location)
#         ibm_db.bind_param(prep_stmt,5,latitude)
#         ibm_db.bind_param(prep_stmt,6,longitude) 
#         ibm_db.execute(prep_stmt)   
#         msg = 'You have successfully registered !'
#         print(msg)
#         return render_template('success.html')




# from mailjet_rest import Client
# import os
# api_key = '****************************1234'
# api_secret = '****************************abcd'
# mailjet = Client(auth=(api_key, api_secret), version='v3.1')
# data = {
#   'Messages': [
#     {
#       "From": {
#         "Email": "209x1a05h6@gprec.ac.in",
#         "Name": "AQI"
#       },
#       "To": [
#         {
#           "Email": "209x1a05h6@gprec.ac.in",
#           "Name": "AQI"
#         }
#       ],
#       "Subject": "Greetings from Mailjet.",
#       "TextPart": "My first Mailjet email",
#       "HTMLPart": "<h3>Dear passenger 1, welcome to <a href='https://www.mailjet.com/'>Mailjet</a>!</h3><br />May the delivery force be with you!",
#       "CustomID": "AppGettingStartedTest"
#     }
#   ]
# }
# result = mailjet.send.create(data=data)
# print result.status_code
# print result.json()
