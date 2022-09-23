import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

df = pd.read_csv(r"CropProps.csv")
print(df.head())


rainfallFeatures = ['RainfallLow', 'RainfallMed', 'RainfallHigh', 'Winter']
for f in rainfallFeatures:
    df[f] = df[f].fillna(False)

le = preprocessing.LabelEncoder()
leM = le.fit(df['Maturity'])

crops = df[['RainfallLow', 'RainfallMed', 'RainfallHigh', 'Winter']]
crops['Maturity'] = leM.transform(df['Maturity'])

cosine_sim = cosine_similarity(crops)

print (cosine_sim.info())