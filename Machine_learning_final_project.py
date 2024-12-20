import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

folder_path = r'C:\Users\Tim Chen\OneDrive\桌面\台大\台大機器學習\html.2023.final.data\release'
rain_path = 'C:/Users/Tim Chen/OneDrive/桌面/台大/台大機器學習/機器學習專案/extend_X_feature/weather/If_rain.xlsx'
holiday_path = 'C:/Users/Tim Chen/OneDrive/桌面/台大/台大機器學習/機器學習專案/extend_X_feature/holiday/If_holiday.xlsx'
all_the_stops_path = 'C:/Users/Tim Chen/OneDrive/桌面/台大/台大機器學習/html.2023.final.data/demographic.json'

with open(all_the_stops_path, 'r', encoding='utf-8') as json_file:
    parsed_data = json.load(json_file)
    all_the_stops = list(parsed_data.keys())
    
def read_feature(feature_path, feature_name):#讀取feature資料，資料我自己編製的excel檔案
    df = pd.read_excel(feature_path)
    df_dict = {}
    for index, row in df.iterrows():
        key = int(row['date'])
        value = row[feature_name]
        df_dict[str(key)] = value
    return df_dict

def read_json_files(folder_path, specific_bike_stop, rain_data, holiday_data):#讀老師給的檔案，怎麼讀的不太重要，因為要看資料的邏輯，所以知道是讀資料就好
    Date = []#注意參數stop_point我指的是讀幾個車站的意思，你要輸出五個車站就打五
    Data = []#rain_data跟holiday_data都是我目前先用很硬幹的方法輸入這個function，你可以參考我的feature然後做一個feature的excel檔案給我，我明天讀

    
    for root, dirs, files in os.walk(folder_path):
        
        if str(root)[-8:][0] != '2':
            continue
        
        for file in files:
            #print(str(file))
            if file.endswith('.json') and str(file)[:9] == specific_bike_stop:
                #print(f'{str(root)[-8:]} :yes')
                file_path = os.path.join(root, file)
  
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    
                    for time, value in data.items():
                        if value:
                            entry = {
                                'date': str(root)[-8:],
                                'bike_stop': str(file)[:9],
                                'time': time,
                                'total': value["tot"],
                                'current': value["sbi"],
                                'can_park': value["bemp"],
                                'open': value["act"],
                                'rain_hour': rain_data[f'{str(root)[-8:]}'],
                                'holiday': holiday_data[f'{str(root)[-8:]}']
                            }
                            Data.append(entry)
        Date.append({str(root)[-8:]})
    
    return Data, Date, 

def time_to_minutes(time_str):#時間是str不能丟train，我改成用總分鐘數，他就會遞增然後是int
    hour, minute = map(int, time_str.split(':'))
    return hour * 60 + minute

def err(y_test, y_pred, total_stop):#老師講義的error function，照打而已
    sum = 0
    for i in range(y_test.size):
        yt = y_test[i]
        yp = y_pred[i]
        sum += 3*(abs(yt-yp)/total_stop)*(abs(yt/total_stop-1/3)+abs(yt/total_stop-2/3))
    sum /= y_test.size
    return sum

best_models = {}

def trainModels(new_data, best_models):
    for stop in new_data:#這裡stop指的是車站名稱
        print("Bike stop ID: ", stop)
        X = []
        y = []
        total_stop = 0
        for elem in new_data[stop]:#這裡開始把資料轉為Xy形式，簡單來說，讓機器看得懂，理論上一個車站X.shape -> (very big, 5), 前面這個5是feature的數量， y.shape -> (very big), 這very big值相同 主要跟你讀進幾個車站的資料成正比
            total_stop = elem.get('total')
            if elem.get('open') == 0:
                continue
            X.append([time_to_minutes(elem.get('time')), elem.get('total'), elem.get('open'), elem.get('rain_hour'), elem.get('holiday')])
            y.append(elem.get('current'))
        X = np.array(X)
        #print(X.shape)
        y = np.array(y)
        X_numeric = X.astype(float)
        smallest_error = float('inf')
        best_model = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=42) #initialize 隨機森林的model
        for i in range(2):#我做十次，用不同的資料切割跑，使用最小的error的那個
            X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.1, random_state=i*42)#資料的切割，這公式很好用 0.1是指切多少當validation，每跑一次用跟前一次不同的亂數種，注意亂數種不用每次compile都不同，因為想比較演算法能不能在同資料下變更好
            
            model = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            #print(y_test)
            #print(y_pred)  
            #for true_value, predicted_value in zip(y_test, y_pred):
                #print(f"True: {true_value}, Predicted: {predicted_value}")
            error = err(y_test, y_pred, total_stop)
            if error < smallest_error:
                smallest_error = error
                best_model = model
        best_models[str(stop)] = best_model
        print(smallest_error)

def transformData(data, specific_bike_stop):
    new_data = {}
    new_data[specific_bike_stop] = []

    for elem in data:#原先資料讀進來是某日->車站->data, 我改成車站->某日->data。簡單來說，是要讓每個車站都有一個model，資料要長這樣比較好讀
        bike_stop = elem.get('bike_stop')
        if bike_stop == specific_bike_stop:
            new_data[bike_stop].append(elem)
    return new_data

rain_data = read_feature(rain_path, "rain_hour")
holiday_data = read_feature(holiday_path, "holiday")
for specific_bike_stop in all_the_stops:
    data, date = read_json_files(folder_path, specific_bike_stop, rain_data, holiday_data)
    new_data = transformData(data, specific_bike_stop)
    trainModels(new_data, best_models)

def find_total_stops(specific_bike_stop):
    data, date = read_json_files(folder_path, specific_bike_stop, rain_data, holiday_data)
    new_data = transformData(data, specific_bike_stop)
    return new_data[specific_bike_stop][1]['total']

import csv
predict_path = 'C:/Users/Tim Chen/OneDrive/桌面/台大/台大機器學習/html.2023.final.data/sample_submission_stage1.csv'
df = pd.read_csv(predict_path)
predict_time = df['id'].tolist()

weather_forecast_path = 'C:/Users/Tim Chen/OneDrive/桌面/台大/台大機器學習/機器學習專案/extend_X_feature/predict_date_rain_hour/predict_rain_hour.xlsx'
pred_rain = read_feature(weather_forecast_path, "rain_hour")
the_holiday = read_feature(weather_forecast_path, "holiday")

my_prediction = []
timer = 0

for elem in predict_time:
    timer += 1
    x_to_pred = []
    future_date = elem[:8]
    stop_to_predict = elem[9:18]
    time_to_predict = time_to_minutes(elem[19:])
        
    rain_hour = pred_rain[future_date]
    holiday = the_holiday[future_date]
    
    x_to_pred.append([time_to_predict, find_total_stops(stop_to_predict), 1, rain_hour, holiday])
    y_pred = best_models[stop_to_predict].predict(x_to_pred)
    my_prediction.append(y_pred)
    
    if timer == 72:
        timer = 0
        print(f'{stop_to_predict}, {future_date}')