from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pickle
import pandas as pd

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # получаем список сгенерированных фичей
    with open('new_features.txt') as f:
        new_features = [(line.strip()) for line in f.readlines()]
    del item['torque']
    # приводим json к требуему виду со всеми фичами
    item['car_name'] = item['name'].split(' ')[0]
    del item['name']
    for i in ['mileage', 'engine', 'max_power']:
        item[i] = float(item[i].split(' ')[0])
    for j in new_features:
        item[j] = 0
    for k in ['fuel', 'seller_type', 
            'transmission', 'owner', 'car_name']:
        if str(k)+'_'+str(item[k]) in list(item.keys()):
            item[f"{k}_{item[k]}"] = 1
        del item[k]
    
    # скалируем фичи
    item_df = pd.json_normalize(item)
    sc=joblib.load('std_scaler.bin')
    
    # считываем модель и делаем предсказание цены
    pickled_model = pickle.load(open('en_model_1.pkl', 'rb'))
    predict = pickled_model.predict(sc.transform(item_df))
    
    return predict


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    # получаем список сгенерированных фичей
    with open('new_features.txt') as f:
        new_features = [(line.strip()) for line in f.readlines()]
    items = items.drop('torque', axis=1)
    
    items['car_name'] = items['name'].str.split(' ').str[0]
    items = items.drop('name', axis=1)
    
    for col in ['mileage', 'engine', 'max_power']:
        items[col] = items[col].str.split(' ').fillna(0).str[0].str.extract(r'(\d+)', expand=False).astype(float)

    for j in new_features:
        items[j] = 0
    for k in ['fuel', 'seller_type', 
            'transmission', 'owner', 'car_name']:
        if str(k)+'_'+str(items[k]) in list(items.columns):
            items[f"{k}_{items[k]}"] = 1
        items = items.drop(k, axis=1)
    
    sc=joblib.load('std_scaler.bin')
    
    # считываем модель и делаем предсказание цены
    pickled_model = pickle.load(open('en_model_1.pkl', 'rb'))
    predict_list = pickled_model.predict(sc.transform(items))
    
    return predict_list