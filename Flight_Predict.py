import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_models():
    folder_path = os.getcwd() + "/Models"
    food_model = pickle.load(open(folder_path+"/Food_Model.sav", 'rb'))           #SEAT_CATEGORY, flighDuration_hour, flight_hour
    insurance_model = pickle.load(open(folder_path+"/Insurance_Model.sav", 'rb')) #geoNetwork_country, SEAT_CATEGORY, flightDuration_hour
    return food_model, insurance_model

def predict_data(flight_info):
    food_model, insurance_model = load_models()
    label_encoder = LabelEncoder()
    flight = pd.DataFrame([flight_info])
    flight["TRIPTYPEDESC"] = label_encoder.fit_transform(flight['TRIPTYPEDESC'])
    flight["SALESCHANNEL"] = label_encoder.fit_transform(flight['SALESCHANNEL'])
    flight["ROUTE"] = label_encoder.fit_transform(flight['ROUTE'])
    flight["geoNetwork_country"] = label_encoder.fit_transform(flight['geoNetwork_country'])
    # SEAT_CATEGORY, flightDuration_hour, flight_hour
    food_predict = food_model.predict([[flight.iloc[0]['SEAT_CATEGORY'], flight.iloc[0]['flightDuration_hour'], flight.iloc[0]['flight_hour']]])

    # geoNetwork_country, SEAT_CATEGORY, flightDuration_hour
    insurance_predict = insurance_model.predict([[flight.iloc[0]['geoNetwork_country'], flight.iloc[0]['SEAT_CATEGORY'], flight.iloc[0]['flightDuration_hour']]])
    dataDict = {"buy_meal": int(food_predict[0]), "buy_insurance": int(insurance_predict[0])}

    return dataDict
    

