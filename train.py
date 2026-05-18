import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv("data/DatasetAristide.csv")

data['heure_min'] = data['heure'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

X = data[['distance_km', 'heure_min']]
y = data['utilise_bus']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open('data/model_bus.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modele entraine et sauvegarde avec succes.")
