import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

df = pd.read_csv(r"CropProps.csv")
print(df.head())


rainfallFeatures = ['RainfallLow', 'RainfallMed', 'RainfallHigh', 'Winter']
for f in rainfallFeatures:
    df[f] = df[f].fillna(False)

df['Maturity'] = df['Maturity'].str.strip()
leM = preprocessing.LabelEncoder()
leM.fit(df['Maturity'])

crops = df[['RainfallLow', 'RainfallMed', 'RainfallHigh', 'Winter']]
crops['Maturity'] = leM.transform(df['Maturity'])

input_LowRainfall = True
input_MedRainfall = False
input_HighRainfall = False
input_Winter = True
input_Maturity = leM.transform(['M'])[0]

inputs = [[ input_LowRainfall, input_MedRainfall, input_HighRainfall, input_Winter, input_Maturity]]

cosine_sim = cosine_similarity(crops, inputs)

results = list(enumerate(cosine_sim)) 
results.sort(key=lambda y: y[1], reverse=True)

for idx in range(5):
    ii = results[idx][0]
    crop = df.iloc[[ii]]
    print(crop)
    
