from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
import folium
import pickle

app = Flask(__name__)

def generer_carte():
    data = pd.read_csv("data/DatasetAristide.csv")
    X = data[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(X)
    couleurs = ['red', 'blue', 'green']
    carte = folium.Map(location=[12.15, 15.05], zoom_start=12)
    for _, row in data.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"{row['eleve']} - {row['quartier']} (Cluster {row['cluster']})",
            icon=folium.Icon(color=couleurs[row['cluster']])
        ).add_to(carte)
    return data, carte._repr_html_()

@app.route('/')
def index():
    data, carte_html = generer_carte()
    eleves = data.to_dict(orient='records')
    total_eleves = len(data)
    bus_users = int(data['utilise_bus'].sum())
    return render_template('index.html', carte=carte_html, eleves=eleves, total_eleves=total_eleves, bus_users=bus_users)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    resultat = None
    conseil = None
    if request.method == 'POST':
        distance = float(request.form['distance'])
        heure = request.form['heure']
        heure_min = int(heure.split(':')[0]) * 60 + int(heure.split(':')[1])
        with open('data/model_bus.pkl', 'rb') as f:
            model = pickle.load(f)
        pred = model.predict([[distance, heure_min]])[0]
        proba = model.predict_proba([[distance, heure_min]])[0]
        if pred == 1:
            resultat = "Cet eleve utilisera probablement le BUS"
            conseil = f"Confiance : {round(proba[1]*100, 1)}%"
        else:
            resultat = "Cet eleve n'utilisera probablement PAS le bus"
            conseil = f"Confiance : {round(proba[0]*100, 1)}%"
    return render_template('prediction.html', resultat=resultat, conseil=conseil)

if __name__ == '__main__':
    app.run(debug=True)
