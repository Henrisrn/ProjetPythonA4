import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

# Set up Spotify API credentials
client_id = '23c4e2e1e192402d82e775935d44042f'
client_secret = 'c864cd4357e7421aae7ff19ac6b6fcea'

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


#%%
# modèle de classification qui prédit le genre musical d'une chanson en se basant sur ses caractéristiques audio.

# Recherche de playlists représentant différents genres musicaux
playlist_ids = {
    'rock': '37i9dQZF1DXcF6B6QPhFDv',
}

# Récupération des caractéristiques audio des chansons dans les playlists
data = {'genre': [], 'tempo': [], 'mode': [], 'energy': [], 'danceability': []}

for genre, playlist_id in playlist_ids.items():
    results = sp.playlist_tracks(playlist_id)
    for track in results['items']:
        try:
            track_uri = track['track']['uri']
            features = sp.audio_features(track_uri)[0]  # Note: access the first element of the returned list
            data['genre'].append(genre)
            data['tempo'].append(features['tempo'])
            data['mode'].append(features['mode'])
            data['energy'].append(features['energy'])
            data['danceability'].append(features['danceability'])
        except Exception as e:
            print("Error with track {}: {}".format(track_uri, str(e)))
            if "429" in str(e):
                # If 429 error, back off for a while before retrying
                time.sleep(30)  # Adjust the sleep duration as needed

# Création d'un DataFrame avec les données
df = pd.DataFrame(data)

# Entraînement du modèle de classification
X = df[['tempo', 'mode', 'energy', 'danceability']]
y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédiction sur de nouvelles données
# Morceau de Daft Punk
track_uri = 'spotify:track:0oks4FnzhNp5QPTZtoet7c'
new_data_features = sp.audio_features(track_uri)
new_data = {
    'tempo': new_data_features[0]['tempo'],
    'mode': new_data_features[0]['mode'],
    'energy': new_data_features[0]['energy'],
    'danceability': new_data_features[0]['danceability'],
}
df_new_data = pd.DataFrame([new_data])
predictions = model.predict(df_new_data)
print(predictions)


#%%

'''
#  modèle de prédiction de popularité des chansons en fonction de certaines caractéristiques audio.
# Recherche de chansons populaires
results = sp.category_playlists(category_id='toplists', country='US', limit=50)
playlist_id = results['playlists']['items'][0]['id']
tracks = sp.playlist_tracks(playlist_id)['items']

# Récupération des caractéristiques audio et de la popularité des chansons
data = {'popularity': [], 'energy': [], 'acousticness': [], 'danceability': []}

for track in tracks:
    features = sp.audio_features(track['track']['uri'])[0]
    data['popularity'].append(track['track']['popularity'])
    data['energy'].append(features['energy'])
    data['acousticness'].append(features['acousticness'])
    data['danceability'].append(features['danceability'])

# Création d'un DataFrame avec les données
df = pd.DataFrame(data)

# Séparation des données en ensembles d'entraînement et de test
X = df[['energy', 'acousticness', 'danceability']]
y = df['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
'''