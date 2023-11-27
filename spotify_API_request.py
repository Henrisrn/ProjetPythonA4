from __future__ import print_function    
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Set up Spotify API credentials
client_id = '23c4e2e1e192402d82e775935d44042f'
client_secret = 'c864cd4357e7421aae7ff19ac6b6fcea'
client_credentials_manager = SpotifyClientCredentials(client_id,client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# retrieve songs ids from playlists
def get_ids(sp,id):
    songs_ids=[]
    res=sp.playlist(id)
    tracks=res['tracks']
    show_tracks(tracks,songs_ids)
    while tracks['next']:
        tracks=sp.next(tracks)
        show_tracks(tracks, songs_ids)
    return songs_ids  
  
def show_tracks(res,uri):
    for i,item in enumerate(res['items']):
        track=item['track']
        uri.append(track['id']) 
        
#extract audio features from song and create dataframe,csv file     
def get_features_df(sp,song_ids):
    features=[]
    idx=0
    while idx < len(song_ids):
        features+=sp.audio_features(song_ids[idx:idx+50])
        idx+=50
    features_ls=[]
    for f in features:
        features_ls.append([f['id'],f['energy'], f['liveness'],f['tempo'], f['speechiness'],f['acousticness'], f['instrumentalness'],
                              f['time_signature'], f['danceability'],f['key'], f['duration_ms'],f['loudness'], f['valence'],f['mode']])
        
    df = pd.DataFrame(features_ls, columns=['id','energy', 'liveness','tempo', 'speechiness','acousticness', 'instrumentalness','time_signature',
                                             'danceability','key', 'duration_ms', 'loudness','valence', 'mode'])
    return df
    




ids=['3ZgmfR6lsnCwdffZUan8EA',
          '1VirHbmy0KMtN8vbaOvIL6',
          '76h0bH2KJhiBuLZqfvPp3K',
          '37i9dQZF1DXaiEFNvQPZrM',
          '37i9dQZF1DXbITWG1ZJKYt',
          '37i9dQZF1DX9RwfGbeGQwP'
          ] 
genres=['pop','rock','r&b','country','jazz','lofi']
i=0
for genre in genres:
    song_ids=get_ids(sp,ids[i])
    song_ids = song_ids[:100]
    df=get_features_df(sp, song_ids)
    df.to_csv('{}_spotify.csv'.format(genre),index=False)
    i+=1
        

