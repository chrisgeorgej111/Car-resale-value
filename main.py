import pickle
import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

st.title("Car Price Prediction")
html_temp = """
   <div style="background-color:#025246 ;padding:10px">
   <h2 style="color:white;text-align:center;">
   Used Car Price Prediction ML App </h2>
   </div>
   """

st.markdown(html_temp, unsafe_allow_html=True)

data1=pd.read_excel('bangalore_cars.xlsx')
data1['new_car_detail'] = data1.new_car_detail.apply(eval)
data1['new_car_overview'] = data1.new_car_overview.apply(eval)
data1['new_car_specs'] = data1.new_car_specs.apply(eval)
reg=[]
for i in data1['new_car_overview']:
    reg.append(i['top'][0]['value'])
ins=[]
for i in data1['new_car_overview']:
    ins.append(i['top'][1]['value'])
ft=[]
for i in data1['new_car_overview']:
    ft.append(i['top'][2]['value'])
seat=[]
for i in data1['new_car_overview']:
    seat.append(i['top'][3]['value'])
kms=[]
for i in data1['new_car_overview']:
    kms.append(i['top'][4]['value'])
model=[]
for i in data1['new_car_detail']:
    model.append(i['model'])
owner=[]
for i in data1['new_car_detail']:
    owner.append(i['ownerNo'])
color=[]
for i in data1['new_car_specs']:
    color.append(i['data'][0]['list'][0]['value'])
et=[]
for i in data1['new_car_specs']:
    et.append(i['data'][0]['list'][1]['value'])
milege=[]
for i in data1['new_car_specs']:
    milege.append(i['top'][0]['value'])
price=[]
for i in data1['new_car_detail']:
    price.append(i['price'])
data1['registration Year']=reg
data1['registration Year']=data1['registration Year'].str.extract(pat='(\d+)', expand=False)
data1['insurance']=ins
data1['model']=model
data1['Engine_type']=et
data1['fuel_type']=ft
data1['no_seats']=seat
data1['no_seats']=data1['no_seats'].str.extract(pat='(\d+)', expand=False)
data1['kms']=kms
data1['kms']=data1['kms'].str.replace(r'[^\d]+', '')
data1['milege']=milege
data1['milege']=data1['milege'].str.extract(pat='(\d+)', expand=False)
data1['color']=color
data1['owner_no']=owner
data1['price']=price
data1['price']=data1['price'].str.extract(pat='(\d+)', expand=False)
data1.drop(['new_car_detail','new_car_overview','new_car_feature','new_car_specs','car_links'],axis=1,inplace=True)

data2=pd.read_excel('chennai_cars.xlsx')
data2['new_car_overview'] = data2.new_car_overview.apply(eval)
data2['new_car_specs'] = data2.new_car_specs.apply(eval)
data2['new_car_detail'] = data2.new_car_detail.apply(eval)
data2['feature'] = data2.new_car_feature.apply(eval)

reg=[]
for i in data2['new_car_overview']:
    reg.append(i['top'][0]['value'])
ins=[]
for i in data2['new_car_overview']:
    ins.append(i['top'][1]['value'])
ft=[]
for i in data2['new_car_overview']:
    ft.append(i['top'][2]['value'])
seat=[]
for i in data2['new_car_overview']:
    seat.append(i['top'][3]['value'])
kms=[]
for i in data2['new_car_overview']:
    kms.append(i['top'][4]['value'])
model=[]
for i in data2['new_car_detail']:
    model.append(i['model'])
owner=[]
for i in data2['new_car_detail']:
    owner.append(i['ownerNo'])
color=[]
for i in data2['new_car_specs']:
    color.append(i['data'][0]['list'][0]['value'])
et=[]
for i in data2['new_car_specs']:
    et.append(i['data'][0]['list'][1]['value'])
milege=[]
for i in data2['new_car_specs']:
    milege.append(i['top'][0]['value'])
price=[]
for i in data2['new_car_detail']:
    price.append(i['price'])

data2['registration Year']=reg
data2['registration Year']=data2['registration Year'].str.extract(pat='(\d+)', expand=False)
data2['insurance']=ins
data2['model']=model
data2['Engine_type']=et
data2['fuel_type']=ft
data2['no_seats']=seat
data2['no_seats']=data2['no_seats'].str.extract(pat='(\d+)', expand=False)
data2['kms']=kms
data2['kms']=data2['kms'].str.replace(r'[^\d]+', '')
data2['milege']=milege
data2['milege']=data2['milege'].str.extract(pat='(\d+)', expand=False)
data2['color']=color
data2['owner_no']=owner
data2['price']=price
data2['price']=data2['price'].str.extract(pat='(\d+)', expand=False)

data2.drop(['new_car_detail','new_car_overview','new_car_feature','new_car_specs','car_links'],axis=1,inplace=True)
data2.drop('feature',axis=1,inplace=True)

data3=pd.read_excel('hyderabad_cars.xlsx')
data3['new_car_overview'] = data3.new_car_overview.apply(eval)
data3['new_car_specs'] = data3.new_car_specs.apply(eval)
data3['new_car_detail'] = data3.new_car_detail.apply(eval)
data3['feature'] = data3.new_car_feature.apply(eval)

reg=[]
for i in data3['new_car_overview']:
    reg.append(i['top'][0]['value'])
ins=[]
for i in data3['new_car_overview']:
    ins.append(i['top'][1]['value'])
ft=[]
for i in data3['new_car_overview']:
    ft.append(i['top'][2]['value'])
seat=[]
for i in data3['new_car_overview']:
    seat.append(i['top'][3]['value'])
kms=[]
for i in data3['new_car_overview']:
    kms.append(i['top'][4]['value'])
model=[]
for i in data3['new_car_detail']:
    model.append(i['model'])
owner=[]
for i in data3['new_car_detail']:
    owner.append(i['ownerNo'])
color=[]
for i in data3['new_car_specs']:
    color.append(i['data'][0]['list'][0]['value'])
et=[]
for i in data3['new_car_specs']:
    et.append(i['data'][0]['list'][1]['value'])
milege=[]
for i in data3['new_car_specs']:
    milege.append(i['top'][0]['value'])
price=[]
for i in data3['new_car_detail']:
    price.append(i['price'])

data3['registration Year']=reg
data3['registration Year']=data3['registration Year'].str.extract(pat='(\d+)', expand=False)
data3['insurance']=ins
data3['model']=model
data3['Engine_type']=et
data3['fuel_type']=ft
data3['no_seats']=seat
data3['no_seats']=data3['no_seats'].str.extract(pat='(\d+)', expand=False)
data3['kms']=kms
data3['kms']=data3['kms'].str.replace(r'[^\d]+', '')
data3['milege']=milege
data3['milege']=data3['milege'].str.extract(pat='(\d+)', expand=False)
data3['color']=color
data3['owner_no']=owner
data3['price']=price
data3['price']=data3['price'].str.extract(pat='(\d+)', expand=False)

data3.drop(['new_car_detail','new_car_overview','new_car_feature','new_car_specs','car_links'],axis=1,inplace=True)
data3.drop('feature',axis=1,inplace=True)

data4=pd.read_excel('kolkata_cars.xlsx')
data4['new_car_overview'] = data4.new_car_overview.apply(eval)
data4['new_car_specs'] = data4.new_car_specs.apply(eval)
data4['new_car_detail'] = data4.new_car_detail.apply(eval)
data4['feature'] = data4.new_car_feature.apply(eval)

reg=[]
for i in data4['new_car_overview']:
    reg.append(i['top'][0]['value'])
ins=[]
for i in data4['new_car_overview']:
    ins.append(i['top'][1]['value'])
ft=[]
for i in data4['new_car_overview']:
    ft.append(i['top'][2]['value'])
seat=[]
for i in data4['new_car_overview']:
    seat.append(i['top'][3]['value'])
kms=[]
for i in data4['new_car_overview']:
    kms.append(i['top'][4]['value'])
model=[]
for i in data4['new_car_detail']:
    model.append(i['model'])
owner=[]
for i in data4['new_car_detail']:
    owner.append(i['ownerNo'])
color=[]
for i in data4['new_car_specs']:
    color.append(i['data'][0]['list'][0]['value'])
et=[]
for i in data4['new_car_specs']:
    et.append(i['data'][0]['list'][1]['value'])
milege=[]
for i in data4['new_car_specs']:
    milege.append(i['top'][0]['value'])
price=[]
for i in data4['new_car_detail']:
    price.append(i['price'])

data4['registration Year']=reg
data4['registration Year']=data4['registration Year'].str.extract(pat='(\d+)', expand=False)
data4['insurance']=ins
data4['model']=model
data4['Engine_type']=et
data4['fuel_type']=ft
data4['no_seats']=seat
data4['no_seats']=data4['no_seats'].str.extract(pat='(\d+)', expand=False)
data4['kms']=kms
data4['kms']=data4['kms'].str.replace(r'[^\d]+', '')
data4['milege']=milege
data4['milege']=data4['milege'].str.extract(pat='(\d+)', expand=False)
data4['color']=color
data4['owner_no']=owner
data4['price']=price
data4['price']=data4['price'].str.extract(pat='(\d+)', expand=False)

data4.drop(['new_car_detail','new_car_overview','new_car_feature','new_car_specs','car_links'],axis=1,inplace=True)
data4.drop('feature',axis=1,inplace=True)

data5=pd.read_excel('jaipur_cars.xlsx')
data5['new_car_overview'] = data5.new_car_overview.apply(eval)
data5['new_car_specs'] = data5.new_car_specs.apply(eval)
data5['new_car_detail'] = data5.new_car_detail.apply(eval)
data5['feature'] = data5.new_car_feature.apply(eval)

reg=[]
for i in data5['new_car_overview']:
    reg.append(i['top'][0]['value'])
ins=[]
for i in data5['new_car_overview']:
    ins.append(i['top'][1]['value'])
ft=[]
for i in data5['new_car_overview']:
    ft.append(i['top'][2]['value'])
seat=[]
for i in data5['new_car_overview']:
    seat.append(i['top'][3]['value'])
kms=[]
for i in data5['new_car_overview']:
    kms.append(i['top'][4]['value'])
model=[]
for i in data5['new_car_detail']:
    model.append(i['model'])
owner=[]
for i in data5['new_car_detail']:
    owner.append(i['ownerNo'])
color=[]
for i in data5['new_car_specs']:
    color.append(i['data'][0]['list'][0]['value'])
et=[]
for i in data5['new_car_specs']:
    et.append(i['data'][0]['list'][1]['value'])
milege=[]
for i in data5['new_car_specs']:
    milege.append(i['top'][0]['value'])
price=[]
for i in data5['new_car_detail']:
    price.append(i['price'])

data5['registration Year']=reg
data5['registration Year']=data5['registration Year'].str.extract(pat='(\d+)', expand=False)
data5['insurance']=ins
data5['model']=model
data5['Engine_type']=et
data5['fuel_type']=ft
data5['no_seats']=seat
data5['no_seats']=data5['no_seats'].str.extract(pat='(\d+)', expand=False)
data5['kms']=kms
data5['kms']=data5['kms'].str.replace(r'[^\d]+', '')
data5['milege']=milege
data5['milege']=data5['milege'].str.extract(pat='(\d+)', expand=False)
data5['color']=color
data5['owner_no']=owner
data5['price']=price
data5['price']=data5['price'].str.extract(pat='(\d+)', expand=False)
data5.drop(['new_car_detail','new_car_overview','new_car_feature','new_car_specs','car_links'],axis=1,inplace=True)
data5.drop('feature',axis=1,inplace=True)
df=pd.concat([data1,data2,data3,data4,data5],ignore_index=True)

df['registration Year'].fillna(df['registration Year'].mode()[0],inplace=True)
df=df.drop(df[df['kms']==''].index)
df['registration Year']=df['registration Year'].astype(int)
df['no_seats']=df['no_seats'].astype(int)
df['kms'] = df['kms'].astype(int)
df['milege']=df['milege'].astype(int)
df['owner_no']=df['owner_no'].astype(int)
df['price']=df['price'].astype(int)
df = df.drop(df[df['kms']==0].index)
df=df.drop(df[df['milege']==0].index)
df = df.drop(df[df['insurance'] == 'Petrol'].index)
df = df.drop(df[df['insurance'] == 'Diesel'].index)
df = df.drop(df[df['insurance'] == '1'].index)
df = df.drop(df[df['insurance'] == '2'].index)
df = df.drop(df[df['insurance'] == 'Electric'].index)
df['tr_no_seats']=np.log(df['no_seats'])
df['tr_kms']=np.log(df['kms'])
df.dropna(inplace=True)
df['tr_milege']=np.log(df['milege'])
df['tr_price']=np.log(df['price'])

X=df[['registration Year','insurance','model','Engine_type','fuel_type','color','owner_no',
     'tr_no_seats', 'tr_kms', 'tr_milege']]
ohe_1 = OneHotEncoder(handle_unknown='ignore')
x_ohe_1 = ohe_1.fit_transform(X[['model']]).toarray()

ohe_2 = OneHotEncoder(handle_unknown='ignore')
x_ohe_2 = ohe_2.fit_transform(X[['color']]).toarray()

ohe_3 = OneHotEncoder(handle_unknown='ignore')
x_ohe_3= ohe_3.fit_transform(X[['insurance']]).toarray()

ohe_4= OneHotEncoder(handle_unknown='ignore')
x_ohe_4 = ohe_4.fit_transform(X[['fuel_type']]).toarray()



X = np.concatenate((df[['registration Year','owner_no',
     'tr_no_seats','tr_kms','tr_milege']].values,x_ohe_1,x_ohe_2,x_ohe_3,x_ohe_4), axis=1)
y=df['tr_price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

rf=RandomForestRegressor()
rf.fit(x_train,y_train)
predi=rf.predict(x_test)
r_2=r2_score(y_test, predi)

mse=mean_squared_error(y_test, predi)

mae=mean_absolute_error(y_test, predi)
with open('regressor.pkl', 'wb') as file:
    pickle.dump(rf, file)

with open('encoder_1.pkl','wb') as file:
    pickle.dump(ohe_1,file)
with open('encoder_2.pkl','wb') as file:
    pickle.dump(ohe_2,file)


with open('encoder_3.pkl','wb') as file:
    pickle.dump(ohe_3,file)

with open('encoder_4.pkl', 'wb') as file:
    pickle.dump(ohe_4, file)

model_year_select=['2015', '2018', '2014', '2020', '2017', '2021', '2019', '2022',
       '2016', '2011', '2009', '2013', '2010', '2008', '2006', '2012',
       '2005', '2007', '2023','2004', '2003', '2002']
ins_select=['Third Party insurance', 'Comprehensive','Zero Dep']
model_select=['Maruti Celerio', 'Ford Ecosport', 'Tata Tiago', 'Hyundai Xcent',
       'Maruti SX4 S Cross', 'Jeep Compass', 'Datsun GO', 'Hyundai Venue',
       'Maruti Ciaz', 'Maruti Baleno', 'Hyundai Grand i10', 'Honda Jazz',
       'Mahindra XUV500', 'Mercedes-Benz GLA', 'Hyundai i20',
       'Tata Nexon', 'Honda City', 'BMW 5 Series', 'Maruti Swift',
       'Renault Duster', 'Mercedes-Benz S-Class', 'Hyundai Santro',
       'Hyundai Santro Xing', 'Mercedes-Benz E-Class', 'Audi A4',
       'Maruti Wagon R', 'Maruti Ertiga', 'Mercedes-Benz C-Class',
       'Toyota Fortuner', 'Hyundai Elantra', 'Audi A6', 'Maruti Alto 800',
       'Mahindra Scorpio', 'Mini 3 DOOR', 'Kia Seltos', 'Maruti Alto',
       'Mercedes-Benz GL-Class', 'Tata New Safari', 'Audi Q7',
       'Renault KWID', 'Hyundai Getz', 'Skoda Rapid', 'Hyundai Creta',
       'Tata Harrier', 'BMW 3 Series GT', 'Renault Lodgy',
       'Skoda Octavia', 'Maruti Ritz', 'Volkswagen Polo',
       'Mahindra KUV 100', 'BMW X3', 'Hyundai i10', 'Volvo S60',
       'Mahindra XUV300', 'MG Hector Plus', 'Honda Brio',
       'Maruti Alto K10', 'Renault Kiger', 'Hyundai EON',
       'Volkswagen Vento', 'Toyota Yaris', 'MG Hector', 'Hyundai Alcazar',
       'Volkswagen T-Roc', 'BMW 3 Series', 'Skoda Superb', 'Audi Q5',
       'Ford Endeavour', 'Ford Figo', 'Maruti Ignis', 'Renault Triber',
       'BMW X5', 'Hyundai Tucson', 'Hyundai Verna', 'Mercedes-Benz GLC',
       'Nissan Terrano', 'Honda CR-V', 'Mercedes-Benz A-Class Limousine',
       'Toyota Innova', 'Hyundai Santa Fe', 'BMW 6 Series',
       'Maruti Baleno RS', 'Renault Captur', 'Maruti Vitara Brezza',
       'Maruti Swift Dzire', 'Fiat Linea', 'Hyundai i20 Active',
       'Honda WR-V', 'Mahindra Ssangyong Rexton', 'Toyota Corolla Altis',
       'Ford Ikon', 'Mitsubishi Cedia', 'Jaguar XF', 'Audi A3',
       'Skoda Kushaq', 'Volkswagen Taigun', 'MG Astor', 'Hyundai Accent',
       'Mercedes-Benz B Class', 'Kia Carnival', 'Skoda Laura', 'BMW X4',
       'Mini Cooper', 'Land Rover Discovery Sport', 'Volvo XC40',
       'Kia Sonet', 'Mahindra Verito', 'Maruti S-Presso',
       'Volkswagen Jetta', 'Datsun RediGO', 'Ford Aspire',
       'Ford Freestyle', 'Audi Q3', 'Tata Tigor', 'Jaguar F-Pace',
       'Mercedes-Benz A Class', 'Toyota Glanza', 'Nissan Magnite',
       'Maruti Gypsy', 'Tata Safari Storme', 'Maruti Celerio X',
       'Mercedes-Benz M-Class', 'Mercedes-Benz GLE',
       'Toyota Urban cruiser', 'Mahindra Thar', 'Mercedes-Benz CLA',
       'Mahindra e2o Plus', 'MG Comet EV', 'Maruti Omni',
       'Volkswagen Tiguan', 'Tata Altroz', 'Tata Nexon EV Max',
       'Tata Indica V2', 'Toyota Innova Crysta', 'Volkswagen Ameo',
       'Tata Nexon EV Prime', 'BMW X1', 'Chevrolet Cruze', 'Toyota Camry',
       'Fiat Punto Abarth', 'Mahindra TUV 300', 'Chevrolet Beat',
       'Maruti Eeco', 'Maruti 1000', 'Citroen C5 Aircross',
       'Mahindra XUV700', 'Hyundai Grand i10 Nios', 'Maruti Zen',
       'Mahindra Quanto', 'Land Rover Freelander 2', 'OpelCorsa',
       'Mahindra Xylo', 'Tata Zest', 'Honda New Accord', 'Skoda Yeti',
       'Maruti SX4', 'Jaguar XE', 'Chevrolet Spark', 'Hyundai i20 N Line',
       'Chevrolet Tavera', 'BMW X7', 'Mahindra Renault Logan',
       'Citroen C3', 'Tata Nano', 'Honda Amaze',
       'Mahindra Bolero Power Plus', 'Tata Manza', 'Maruti Esteem',
       'Tata Hexa', 'Nissan Micra Active', 'Mitsubishi Lancer',
       'Ford Fiesta', 'Mahindra Bolero Camper', 'Fiat Punto',
       'Kia Carens', 'Chevrolet Enjoy', 'Volkswagen Tiguan Allspace',
       'Skoda Slavia', 'Mahindra Marazzo', 'Tata Indigo', 'Jaguar XJ',
       'Skoda Fabia', 'Tata Sumo', 'Ford Mondeo', 'Nissan Sunny',
       'Fiat Palio', 'Toyota Etios', 'Maruti Estilo', 'Mahindra Bolero',
       'Jeep Meridian', 'BMW 1 Series', 'Volvo XC 90',
       'Audi A3 cabriolet', 'MG Gloster', 'Land Rover Range Rover Sport',
       'Nissan Micra', 'Fiat Punto EVO', 'Mini Cooper Countryman',
       'Renault Fluence', 'Maruti A-Star', 'Tata Nexon EV',
       'Chevrolet Sail', 'BMW 7 Series', 'Maruti XL6', 'Hyundai Sonata',
       'Honda Civic', 'Maruti Ertiga Tour', 'Mercedes-Benz GLS',
       'Isuzu MU 7', 'Maruti 800', 'Hyundai Aura',
       'BMW 3 Series Gran Limousine', 'Volvo S90', 'Tata Indica',
       'Tata Punch', 'Honda BR-V', 'Mahindra Scorpio N', 'Skoda Kodiaq',
       'Tata Tiago NRG', 'Datsun GO Plus', 'BMW 2 Series',
       'Maruti Wagon R Stingray', 'Mini 5 DOOR', 'Fiat Grande Punto',
       'Mahindra KUV 100 NXT', 'Mercedes-Benz GLA Class',
       'Chevrolet Aveo', 'Land Rover Range Rover Velar', 'Toyota Hyryder',
       'Maruti Zen Estilo', 'Toyota Etios Liva',
       'Land Rover Range Rover Evoque', 'Maruti Versa', 'Isuzu MU-X',
       'Fiat Punto Pure', 'Honda Mobilio', 'Chevrolet Optra',
       'Volvo S 80', 'Mitsubishi Pajero', 'Audi A8', 'Volvo XC60',
       'Mercedes-Benz AMG GLA 35', 'Mercedes-Benz AMG A 35',
       'Volkswagen Virtus', 'Land Rover Discovery', 'Lexus ES', 'Audi Q2',
       'Nissan Kicks', 'Mahindra TUV 300 Plus', 'Maruti Grand Vitara',
       'Toyota Etios Cross', 'Mahindra Alturas G4', 'Mahindra Jeep',
       'Toyota Qualis', 'Maruti Swift Dzire Tour', 'Volkswagen Passat',
       'Ford Fiesta Classic', 'Maruti Brezza', 'Land Rover Range Rover',
       'Fiat Avventura', 'Renault Scala', 'Honda City Hybrid',
       'Tata Aria', 'Volvo V40', 'Tata Bolt', 'MG ZS EV',
       'Mahindra E Verito', 'Hyundai Xcent Prime', 'Mercedes-Benz EQC',
       'Fiat Abarth Avventura', 'Hindustan Motors Contessa',
       'Mahindra Bolero Neo', 'Tata Yodha Pickup', 'Tata Indigo Marina',
       'Chevrolet Captiva', 'Mahindra Bolero Pik Up Extra Long',
       'Toyota Corolla', 'Jeep Wrangler', 'Ambassador',
       'Volvo S60 Cross Country', 'Jeep Compass Trailhawk',
       'Tata Sumo Victa', 'Porsche Macan', 'Porsche Panamera',
       'Mercedes-Benz AMG GT', 'Audi S5 Sportback', 'Renault Pulse',
       'Jaguar F-TYPE', 'Tata Tigor EV', 'Toyota Fortuner Legender',
       'Mercedes-Benz AMG GLC 43', 'Chevrolet Aveo U-VA', 'Hyundai Kona',
       'Isuzu D-Max', 'Porsche 911', 'Volkswagen CrossPolo',
       'Maruti Jimny']
fuel_type=['Petrol', 'Diesel', 'LPG', 'CNG','Electric']
color_select=['White', 'Red', 'Others', 'Gray', 'Grey', 'Maroon', 'Orange',
       'Silver', 'Blue', 'Brown', 'Yellow', 'Black', 'Golden', 'Green',
       'O Purple', 'Other', 'Gold', 'TITANIUM GREY', 'Violet',
       'MODERN STEEL METALLIC', 'PLATINUM WHITE', 'Golden Brown',
       'Aurora Black Pearl', 'Beige', 'Star Dust', 'Flash Red', 'Purple',
       'PLATINUM WHITE PEARL', 'Wine Red', 'Taffeta White',
       'Minimal Grey', 'Fiery Red', 'T Wine', 'Prime Star Gaze',
       'TAFETA WHITE', 'P Black', 'Golden brown', 'METALL',
       'MET ECRU BEIGE', 'COPPER', 'TITANIUM', 'CHILL', 'Burgundy',
       'Lunar Silver Metallic', 'SILKY SILVER', 'BERRY RED',
       'PREMIUM AMBER METALLIC', 'R EARTH', 'PLATINUM SILVER',
       'ORCHID WHITE PEARL', 'CARNELIAN RED PEARL', 'POLAR WHITE',
       'BEIGE', 'Hip Hop Black', 'Nexa Blue', 'Passion Red',
       'Cirrus White', 'Arizona Blue', 'Galaxy Blue', 'Silky Silver',
       'Modern Steel Metal', 'GOLDEN BROWN', 'Polar White',
       'Burgundy Red Metallic', 'magma gray', 'CBeige', 'Goldan BRWOON',
       'm grey', 'b red', 'Granite Grey', 'urban titanim', 'g brown',
       'beige', 'Rosso Brunello', 'a silver', 'b grey', 'Radiant Red M',
       'c bronze', 'Champagne Mica Metallic', 'Bold Beige Metallic',
       'Starry Black', 'Sleek Silver', 'Symphony Silver', 'Phantom Black',
       'Metallic Magma Grey', 'c brown', 'chill',
       'Metallic Glistening Grey', 'Superior white',
       'Modern Steel Metallic', 'Silky silver', 'Arctic Silver',
       'Urban Titanium Metallic', 'Smoke Grey', 'Pearl Arctic White',
       'Foliage', 'Sky Blue', 'Off White', 'Bronze', 'G Brown', 'Parpel',
       'Outback Bronze', 'Cherry Red', 'Sunset Red', 'Silicon Silver',
       'golden brown', 'Dark Blue',
       'Light Silver', 'Out Back Bronze', 'Bright Silver',
       'Porcelain White', 'Tafeta White', 'Coral White', 'Diamond White',
       'Brick Red', 'Carnelian Red Pearl', 'Mediterranean Blue',
       'Mist Silver', 'Gravity Gray', 'Candy White',
       'Metallic Premium silver', 'Glistening Grey', 'Super white',
       'Deep Black Pearl', 'Twilight Blue', 'Caviar Black',
       'Pearl Met. Arctic White', 'Pearl White', 'Metallic silky silver',
       'Pure white', 'StarDust', 'Alabaster Silver Metallic - Amaze',
       'Ray blue', 'Glacier White Pearl', 'OUTBACK BRONZE',
       'Solid Fire Red', 'Daytona Grey', 'Metallic Azure Grey',
       'Moonlight Silver', 'Fire Brick Red', 'Cashmere',
       'Pearl Snow White', 'Light Orange']
no_seats_select=[5,6,7,8,9,10]
owner_no=[3, 2, 1, 4, 5]

with st.form("my_form"):
    col1, col2 = st.columns([3,3])
    with col1:
        st.write(' ')
        model_year = st.selectbox("model_year", sorted(model_year_select), key=1)
        ins= st.selectbox("insurance",ins_select , key=2)
        model_name=st.selectbox("model_name",model_select, key=3)
        fuel_type = st.selectbox("fuel_type",fuel_type , key=4)
        color= st.selectbox("color", sorted(color_select), key=5)
        no_seats=st.selectbox("no_seats", sorted(no_seats_select), key=6)
        owner_no=st.selectbox("owner_no", sorted(owner_no),key=7)

    with col2:
        kms=st.text_input("Enter kms (Min:1000 & Max:550000)")
        mileage=st.text_input("Enter Mileage(kmpl)(Min:7 & Max:23)")

        submit = st.form_submit_button(label="PREDICT RESALE PRICE")

    flag = 0
    pattern = '[0-9]*\.?[0-9]+'
    for i in [kms,mileage]:
        if re.match(pattern, i):
            pass
        else:
            flag = 1
            break

if submit and flag == 1:
    if len(i) == 0:
        st.write("please enter a valid number space not allowed")
    else:
        st.write("You have entered an invalid value: ", i)

if submit and flag == 0:
    if submit and flag == 0:
        with open(r"regressor.pkl", 'rb') as file:
            loaded_model = pickle.load(file)

        with open(r"encoder_1.pkl", 'rb') as f:
            oh_1_load = pickle.load(f)

        with open(r"encoder_2.pkl", 'rb') as f:
            oh_2_load = pickle.load(f)

        with open(r"encoder_3.pkl", 'rb') as f:
            oh_3_load = pickle.load(f)

        with open(r"encoder_4.pkl", 'rb') as f:
            oh_4_load = pickle.load(f)

        new_sample = np.array([[np.log(float(kms)),np.log(float(mileage)),float(no_seats),model_year,owner_no,ins,model_name,fuel_type,
                                color]])
        new_sample_ohe_1 = oh_1_load.transform(new_sample[:, [6]]).toarray()
        new_sample_ohe_2 = oh_2_load.transform(new_sample[:, [8]]).toarray()
        new_sample_ohe_3=oh_3_load.transform(new_sample[:, [5]]).toarray()
        new_sample_ohe_4 =oh_4_load.transform(new_sample[:, [7]]).toarray()

        new_sample =np.concatenate((new_sample[:, [0, 1, 2,3,4]], new_sample_ohe_1, new_sample_ohe_2,new_sample_ohe_3,new_sample_ohe_4),axis=1)
        new_pred = loaded_model.predict(new_sample)[0]
        st.write('## :green[Predicted car resale Price(in lacs):] ', round(np.exp(new_pred)))









