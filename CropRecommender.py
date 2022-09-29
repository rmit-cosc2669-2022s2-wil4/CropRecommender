import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

df = pd.read_csv(r"CropProps.csv")
#  print(df.head())

rainfallFeatures = ['RainfallLow', 'RainfallMed', 'RainfallHigh', 'Winter']
for f in rainfallFeatures:
    df[f] = df[f].fillna(False)

df['Maturity'] = df['Maturity'].str.strip()
leM = preprocessing.LabelEncoder()
leM.fit(df['Maturity'])

crops = df[['RainfallLow', 'RainfallMed', 'RainfallHigh', 'Winter']]
crops['Maturity'] = leM.transform(df['Maturity'])

# Get user's input 
print("Low Rainfall (<350mm)?")
input_LowRainfall = bool(input())

print("Medium Rainfall (350mm < 500MM)?")
input_MedRainfall = bool(input())

print("High Rainfall (>500mm)?")
input_HighRainfall = bool(input())

print("Winder Crop?")
input_Winter = bool(input())

print("Maturity (" + ",".join(leM.classes_) + ")?")
input_Maturity = leM.transform([input()])[0]

inputs = [[ input_LowRainfall, input_MedRainfall, input_HighRainfall, input_Winter, input_Maturity]]
# inputs = [[ True, False, False, False, 2]]

cosine_sim = cosine_similarity(crops, inputs)

results = list(enumerate(cosine_sim)) 
results.sort(key=lambda y: y[1], reverse=True)

fmt = '{:15} {:20} {:11} {:13} {:13} {:13} {:8} {:8}'

print(fmt.format('Name','Type','Max Quality','Low Rainfall','Med Rainfall','High Rainfall','Winter','Maturity'))

for idx in range(5):
    ii = results[idx][0]
    crop = df.iloc[[ii]]
    # print(crop)

    name = df.iloc[ii]['Name']
    type = df.iloc[ii]['Type']
    maxQuality  = df.iloc[ii]['MaxQuality']
    rainfallLow = 'Yes' if df.iloc[ii]['RainfallLow'] else 'No'
    rainfallMed = 'Yes' if df.iloc[ii]['RainfallMed'] else 'No'
    rainfallHigh = 'Yes' if df.iloc[ii]['RainfallHigh'] else 'No'
    winter = 'Yes' if df.iloc[ii]['Winter'] else 'No'
    maturity = df.iloc[ii]['Maturity']

    print(fmt.format(name, type, maxQuality, rainfallLow, rainfallMed, rainfallHigh, winter, maturity))
   