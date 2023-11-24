import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os
import plotly.express as px
from bokeh.plotting import figure
from bokeh.models import HoverTool
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from random import randint
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.impute import SimpleImputer
import re


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

st.set_page_config(layout="wide")


#%% MISE EN PLACE DE LA SESSION STREAMLIT POUR POUVOIR CHARGER ET ENTRAINER LE MODEL

@st.cache_data
def load_and_train():
    if 'data' not in st.session_state or 'model' not in st.session_state:
        #%% TRAITEMENT DE LA DONNEE
        columns_to_load = ["name","artists","daily_rank","daily_movement","weekly_movement","country","snapshot_date","popularity","is_explicit","duration_ms","album_name","album_release_date","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","time_signature"] 
        st.session_state['data'] = pd.read_csv("universal_top_spotify_songs.csv", usecols=columns_to_load,sep=",")
        st.session_state['features'] = st.session_state['data'][['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'mode']]
        st.session_state['target'] = np.random.choice(['Rock', 'Pop', 'Jazz', 'Hip-Hop'], size=len(st.session_state['data']))
        
        for colgraph in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']:
            st.session_state['data'][colgraph] = st.session_state['data'][colgraph].astype(str).str.replace(',', '.').astype(float)
        st.session_state['data']['snapshot_date'] = pd.to_datetime(st.session_state['data']['snapshot_date'])
        st.session_state['data']['album_release_date'] = pd.to_datetime(st.session_state['data']['album_release_date'])

        #country_codes = {'SK': 'South Korea', 'CA': 'Canada', 'ES': 'Spain', 'PR': 'Puerto Rico','ZA':'South Africa'}
        #st.session_state['data']['country'] = st.session_state['data']['country'].map(country_codes)
        
        
        #%%ENTRAINEMENT DU MODEL
        features = st.session_state['data'][['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'mode']]
        target = np.random.choice(['Rock', 'Pop', 'Jazz', 'Hip-Hop'], size=len(st.session_state['data']))
        with st.spinner('Training model...'):
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            # Standardize features
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            # Initialize and train your model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print("Accuracy:", accuracy_score(y_test, y_pred))
        st.session_state['clf']=model
        st.session_state['scaler'] =scaler
        st.session_state['data']['genre'] = st.session_state['clf'].predict(st.session_state['scaler'].transform(st.session_state['features']))

def genre_prediction_model():
    # Création de dataframes par genre
    df_pop=pd.read_csv('pop_spotify.csv')
    df_rock=pd.read_csv('rock_spotify.csv')
    df_country=pd.read_csv('country_spotify.csv')
    df_rb=pd.read_csv('r&b_spotify.csv')

    # scale the data
    col_names=df_pop.drop(['id','time_signature'],axis=1).columns
    sc=MinMaxScaler()
    df_pop=pd.DataFrame(sc.fit_transform(df_pop.drop(['id','time_signature'],axis=1)),columns=col_names)
    df_rock=pd.DataFrame(sc.fit_transform(df_rock.drop(['id','time_signature'],axis=1)),columns=col_names)
    df_rb=pd.DataFrame(sc.fit_transform(df_rb.drop(['id','time_signature'],axis=1)),columns=col_names)
    df_hiphop=pd.DataFrame(sc.fit_transform(df_country.drop(['id','time_signature'],axis=1)),columns=col_names)
    # attribute a number to each genre
    df_pop['genre']=1
    df_rock['genre']=2
    df_rb['genre']=3
    df_country['genre']=4

    # create a concatenated dataframe
    df=pd.concat([df_pop, df_rock, df_rb, df_country])

    X = df.drop(['genre', 'id','time_signature'], axis=1)
    y = df['genre']
    # Replace NaN values with the mean of each column
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Entraînement du modèle de classification
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def predict_genre_song(track_uri, sp,genre_predict_model):
    track_uri = extract_track_id_from_uri(track_uri)
    print(track_uri)
    new_data_features = sp.audio_features(track_uri)
    # extracted features
    new_data = pd.DataFrame({
        'energy': [new_data_features[0]['energy']],
        'liveness': [new_data_features[0]['liveness']],
        'tempo': [new_data_features[0]['tempo']],
        'speechiness': [new_data_features[0]['speechiness']],
        'acousticness': [new_data_features[0]['acousticness']],
        'instrumentalness': [new_data_features[0]['instrumentalness']],
        'danceability': [new_data_features[0]['danceability']],
        'key': [new_data_features[0]['key']],
        'duration_ms': [new_data_features[0]['duration_ms']],
        'loudness': [new_data_features[0]['loudness']],
        'valence': [new_data_features[0]['valence']],
        'mode': [new_data_features[0]['mode']],
    })
    # Make predictions using the trained model
    new_predictions_proba = genre_predict_model.predict_proba(new_data)
    class_labels = genre_predict_model.classes_
    genre_mapping = {1: 'Pop', 2: 'Rock', 3: 'R&B', 4: 'Country'}
    
    # Get the predicted genre label
    predicted_genre_label = class_labels[np.argmax(new_predictions_proba)]

    # Display the predicted genre using the mapping
    predicted_genre = genre_mapping[predicted_genre_label]
    # Format and display the probability scores as percentages
    proba_percentage = [f"{genre_mapping[label]}: {proba*100:.2f}%" for label, proba in zip(class_labels, new_predictions_proba[0])]
    return f'Predicted Genre: {predicted_genre}', f'Probability Scores:\n {"  ,  ".join(proba_percentage)}'
        
        
def extract_track_id_from_uri(spotify_uri):
    # Find the start and end indices of the track ID
    start_index = spotify_uri.find('/track/') + len('/track/')
    end_index = spotify_uri.find('?', start_index)

    # Extract the track ID using slicing
    track_id = spotify_uri[start_index:end_index]

    # Return the transformed URI
    return f'spotify:track:{track_id}' if track_id else None


def plot_valence_popularity_regression(data, colgraph=st, coloptions=st.sidebar):
    fig = px.scatter(data, x='valence', y='popularity', trendline='ols', title='Valence et Popularité')
    colgraph.plotly_chart(fig)
    

def plot_time_signature_distribution(data, colgraph=st, coloptions=st.sidebar):
    fig = px.bar(data['time_signature'].value_counts().reset_index(), x='index', y='time_signature', title='Répartition des Signatures Temporelles')
    colgraph.plotly_chart(fig)


def plot_explicit_proportion(data, colgraph=st, coloptions=st.sidebar):
    fig = px.pie(data, names='is_explicit', title='Proportion de Morceaux Explicites')
    colgraph.plotly_chart(fig)


def plot_common_key(data, colgraph=st, coloptions=st.sidebar):
    fig = px.bar(data['key'].value_counts().reset_index(), x='index', y='key', title='Clé Musicale la Plus Commune')
    colgraph.plotly_chart(fig)


def plot_valence_vs_duration(data, colgraph=st, coloptions=st.sidebar):
    fig = px.scatter(data, x='valence', y='duration_ms', title='Valence contre Durée')
    colgraph.plotly_chart(fig)

def plot_liveness_distribution(data, colgraph=st, coloptions=st.sidebar):
    fig = px.histogram(data, x='liveness', title='Distribution de la Liveness')
    colgraph.plotly_chart(fig)


# Note: Ceci nécessite que la colonne 'genre' soit présente dans vos données
def plot_genre_distribution(data, colgraph=st, coloptions=st.sidebar):
    if 'genre' in data.columns and not data['genre'].isnull().all():
        fig = px.bar(data['genre'].value_counts().reset_index(), x='index', y='genre', title='Genres Musicaux les Plus Communs')
        colgraph.plotly_chart(fig)
    else:
        colgraph.error("The 'genre' column is missing or empty.")



def plot_mode_distribution(data, colgraph=st, coloptions=st.sidebar):
    fig = px.pie(data, names='mode', title='Répartition des Modes (Majeur/Mineur)')
    colgraph.plotly_chart(fig)


    
def plot_feature_correlation(data, colgraph=st, coloptions=st.sidebar):
    corr = data[['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title='Corrélation entre Caractéristiques Musicales')
    colgraph.plotly_chart(fig)


def plot_loudness_boxplot(data, colgraph=st, coloptions=st.sidebar):
    fig = px.box(data, y='loudness', title='Niveaux de Loudness')
    colgraph.plotly_chart(fig)

    
    
def plot_popularity_distribution(data, colgraph=st, coloptions=st.sidebar):
    # Use a sidebar slider to select the number of bins
    bins = coloptions.slider('Select number of bins for popularity distribution', 5, 50, 20)
    fig = px.histogram(data, x='popularity', nbins=bins, title='Distribution de la Popularité')
    colgraph.plotly_chart(fig)
    

# Energy by Country with Sorting Selector
def plot_energy_by_country(data, colgraph=st, coloptions=st.sidebar):
    # Use a sidebar radio button to select the sort order
    sort_order = coloptions.radio('Select sort order for energy by country', ['Ascending', 'Descending'])
    energy_country = data.groupby('country')['energy'].mean().reset_index()
    energy_country = energy_country.sort_values(by='energy', ascending=(sort_order == 'Ascending'))
    fig = px.bar(energy_country, y='country', x='energy', orientation='h', title='Energie Moyenne par Pays')
    colgraph.plotly_chart(fig)

# Danceability vs Energy with Alpha Selector
def plot_danceability_vs_energy(data, colgraph=st, coloptions=st.sidebar):
    alpha = coloptions.slider('Select alpha for danceability vs energy plot', 
                              min_value=0.1, max_value=1.0, value=0.5, key='alpha_slider')
    fig = px.scatter(data, x='danceability', y='energy', opacity=alpha,
                     title='Danceabilité contre Energie', labels={'danceability': 'Danceabilité', 'energy': 'Energie'})
    colgraph.plotly_chart(fig)

def plot_duration_distribution(data, colgraph=st, coloptions=st.sidebar):
    unit = coloptions.radio('Select duration unit', ('Minutes', 'Milliseconds'), key='duration_unit')
    bins = coloptions.slider('Select number of bins for duration distribution', min_value=5, max_value=50, value=20, key='duration_bins')
    durations = data['duration_ms'] if unit == 'Milliseconds' else data['duration_ms'] / 60000
    title = 'Distribution des Durées des Morceaux' if unit == 'Milliseconds' else 'Distribution des Durées des Morceaux (en Minutes)'
    fig = px.histogram(data, x=durations, nbins=bins, title=title)
    fig.update_layout(xaxis_title=f'Durée ({unit})', yaxis_title='Nombre de Morceaux')
    colgraph.plotly_chart(fig)


def filter_data(data, selected_countries, selected_daily_movement, selected_date_range, selected_explicit):
    # Filter data based on user input
    filtered_data = data[(data['country'].isin(selected_countries)) & 
                        # (data['snapshot_date'].isin(selected_snapshot_dates)) & 
                        (data['daily_movement'].between(*selected_daily_movement)) &
                        (data['snapshot_date'].between(*selected_date_range))]

    if selected_explicit == 'Explicit Content':
        filtered_data = filtered_data[filtered_data['is_explicit']]
    elif selected_explicit == 'Non-Explicit Content':
        filtered_data = filtered_data[~filtered_data['is_explicit']]

    return filtered_data



def plot_data_filtered(filtered_data, colgraph=st, coloptions=st.sidebar):
    # Display filtered data in a table
    colgraph.header('Filtered Data')
    colgraph.write(f'Data Dimension: {filtered_data.shape[0]} rows and {filtered_data.shape[1]} columns.')
    colgraph.dataframe(filtered_data)


def visualisation(data):
    country_counts = data['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    country_counts['count'] = country_counts['count'] - 1200
    country_counts = country_counts[country_counts['count'] > 0]
    # Initialize the map
    with open('custom.geo.json', 'r', encoding='utf-8') as f:
        geo_json_data = json.load(f)
    # Initialize the map
    m = folium.Map(location=[20, 0], zoom_start=2)
    folium.GeoJson(geo_json_data).add_to(m)
    # Add the choropleth layer
    folium.Choropleth(
        geo_data=geo_json_data,
        name='choropleth',
        data=country_counts,
        columns=['country', 'count'],
        key_on='feature.properties.postal', 
        fill_color='YlGnBu',
        fill_opacity=0.5,
        line_opacity=0.1,
        legend_name='Song frequency'
    ).add_to(m)

    try:
        # Add user input for data filtering
        st.header('Data Filtering')

        # Set default values for multiselect for countries
        default_countries = ['FR', 'US', 'SK']
        selected_countries = st.multiselect('Select Country', data['country'].unique(), default=default_countries)

        # Set default values for multiselect for snapshot_date
        # default_last_dates = list(data['snapshot_date'].nlargest(10))
        # selected_snapshot_dates = st.sidebar.multiselect('Select Snapshot Dates', sorted(data['snapshot_date'].unique()), default=default_last_dates)
        # selected_snapshot_dates = st.sidebar.slider('Select Snapshot Dates', min_value=data['snapshot_date'].min(), max_value=data['snapshot_date'].max(), value=(data['snapshot_date'].min(), data['snapshot_date'].max()))
        min_date = data['snapshot_date'].min()
        max_date = data['snapshot_date'].max()
        default_date_range = [min_date, max_date]
        selected_date_range = st.select_slider('Select Snapshot Date Range', options=pd.date_range(min_date, max_date, freq='D'), value=default_date_range)

        selected_daily_movement = st.slider('Select Daily Movement Range', min_value=data['daily_movement'].min(), max_value=data['daily_movement'].max(), value=(data['daily_movement'].min(), data['daily_movement'].max()))

        # Create mutually exclusive selection for explicitness
        explicit_options = ['Both', 'Explicit Content', 'Non-Explicit Content']
        selected_explicit = st.radio('Select Explicitness', explicit_options)

        filtered_data = filter_data(data, selected_countries, selected_daily_movement, selected_date_range, selected_explicit)
        
        plot_data_filtered(filtered_data)
    except:
        st.write("Problem with data_filtered")

    
    # if col1.button("world map"):   
    folium_static(m)
    # st.sidebar.header('Data Filtering by fast Henri (=jugement GPT)')

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='popularity_distribution')
        plot_popularity_distribution(filtered_data, col1, col2) if on else plot_popularity_distribution(data, col1, col2)
    except:
        col1.write("Problem with popularity_distribution graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='energy_by_country')
        plot_energy_by_country(filtered_data, col1, col2) if on else plot_energy_by_country(data, col1, col2)
    except:
        col1.write("Problem with energy_by_country graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='danceability_vs_energy')
        plot_danceability_vs_energy(filtered_data, col1, col2) if on else plot_danceability_vs_energy(data, col1, col2)
    except:
        col1.write("Problem with danceability_vs_energy graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='duration_distribution')
        plot_duration_distribution(filtered_data, col1, col2) if on else plot_duration_distribution(data, col1, col2)
    except:
        col1.write("Problem with duration_distribution graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='loudness_boxplot')
        plot_loudness_boxplot(filtered_data, col1, col2) if on else plot_loudness_boxplot(data, col1, col2)
    except:
        col1.write("Problem with loudness_boxplot graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='feature_correlation')
        plot_feature_correlation(filtered_data, col1, col2) if on else plot_feature_correlation(data, col1, col2)
    except:
        col1.write("Problem with feature_correlation graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='mode_distribution')
        plot_mode_distribution(filtered_data, col1, col2) if on else plot_mode_distribution(data, col1, col2)
    except:
        col1.write("Problem with mode_distribution graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='genre_distribution')
        plot_genre_distribution(filtered_data, col1, col2) if on else plot_genre_distribution(data, col1, col2)
    except:
        col1.write("Problem with genre_distribution graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='liveness_distribution')
        plot_liveness_distribution(filtered_data, col1, col2) if on else plot_liveness_distribution(data, col1, col2)
    except:
        col1.write("Problem with liveness_distribution graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='valence_vs_duration')
        plot_valence_vs_duration(filtered_data, col1, col2) if on else plot_valence_vs_duration(data, col1, col2)
    except:
        col1.write("Problem with valence_vs_duration graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='common_key')
        plot_common_key(filtered_data, col1, col2) if on else plot_common_key(data, col1, col2)
    except:
        col1.write("Problem with common_key graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='explicit_proportion')
        plot_explicit_proportion(filtered_data, col1, col2) if on else plot_explicit_proportion(data, col1, col2)
    except:
        col1.write("Problem with explicit_proportion graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='time_signature_distribution')
        plot_time_signature_distribution(filtered_data, col1, col2) if on else plot_time_signature_distribution(data, col1, col2)
    except:
        col1.write("Problem with time_signature_distribution graph")

    col1, col2 = st.columns([3, 1])
    try:
        on = col2.toggle('Use filtered data', key='valence_popularity_regression')
        plot_valence_popularity_regression(filtered_data, col1, col2) if on else plot_valence_popularity_regression(data, col1, col2)
    except:
        col1.write("Problem with valence_popularity_regression graph")
    
    data['popularity_moving_average'] = data['popularity'].rolling(window=7).mean()

    # Create a Bokeh chart (e.g., popularity trend)
    p = figure(title='Popularity Trend', x_axis_label='Date', y_axis_label='Popularity Moving Average', x_axis_type='datetime')
    p.line(data['snapshot_date'], data['popularity_moving_average'], legend_label='Popularity Trend', line_width=2)
    p.add_tools(HoverTool(tooltips=[('Date', '@x{%F}'), ('Popularity', '@y')], formatters={'@x': 'datetime'}))

    # Display the Bokeh chart in Streamlit
    # if st.button("bokeh_chart"):   
    col1.bokeh_chart(p, use_container_width=True)

    

def home(sp):
    genre_predict_model = genre_prediction_model()
    st.title('Spotify Data Analysis')
    
    # Using column layout for better control
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image('logoSpotify.png', width=200)
    
    with col2:
        st.subheader('Welcome to the Spotify Data Analysis App')
        st.write('''
        Dive into the fascinating world of music and discover patterns, 
        trends, and insights from Spotify's top tracks.
        ''')
    
    # Example of a dataframe snippet
    st.write('Here is a glimpse into the dataset we are analyzing:')
    st.dataframe(st.session_state['data'].head())

    # Team presentation
    expander_bar = st.expander("Meet the Team")
    # st.subheader('Meet the Team')
    expander_bar.markdown('''
    - **Henri Serano**: Le fast guy, mangeur de crêpes
    - **Sara Thibierge**: La putchiste, faiseuse de crêpes
    - **Eloi Seidlitz**: Le pésident déchu, mangeur de crêpes
    ''')

    # You could also use st.image or st.markdown to add images of team members
    
    # For animations and more complex styling, you would need to inject custom HTML/CSS
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("starry-sky-8199764_1280.jpg");
            background-size: cover;
        }
        .big-font {
            font-size:300% !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title('Want to Predict Your Music Genre?')
    # Get user input for Spotify track URI
    track_uri = st.text_input('Enter Spotify song link ex: https://open.spotify.com/intl-fr/track/0oks4FnzhNp5QPTZtoet7c?si=5e6ea8bf8d3746fa ')

    # Add a button to trigger genre prediction
    if st.button('Predict Genre'):
        if track_uri is not "": 
            # Use your existing code to make predictions
            # Extract audio features for the new track using Spotify API
            predicted_genre, proba_score = predict_genre_song(track_uri, sp, genre_predict_model)
            st.success(predicted_genre)
            st.success(proba_score)
     
    
    # Using custom HTML to inject styles or even animations
    st.markdown(
        '<p class="big-font">Spotify Data Rocks!</p>', 
        unsafe_allow_html=True
    )
    
# Fonction pour afficher les statistiques

def statistics(results):
    st.title('Statistiques des Données Spotify')
    st.write('Voici un résumé des statistiques clés issues de l\'analyse des données Spotify:')

    st.metric("Moyenne de popularité",results['Mean Popularity'])
    st.metric("Durée médiane (ms)",results['Median Duration (ms)'])
    st.metric("Signature temporelle la plus courante", results['Mode Time Signature'])
    st.metric("Fourchette de Loudness (dB)",results['Range Loudness (dB)'])
        

    st.metric("Tempo STD (BPM)",results['STD Tempo (BPM)'])
    st.metric("Corrélation Danse/Energie",results['Correlation Danceability and Energy'])
    st.metric("Jours moyens entre cliché et sortie d'album",results['Mean Days between Snapshot and Album Release'])
    st.metric("Proportion de morceaux > 0.7 en danse",results['Percentage Danceable Tracks > 0.7'])
        
    st.metric("Clé la plus commune", results['Most Common Key'])
    st.metric("Différence de Loudness par explicité",results['Difference Mean Loudness Explicitness'])
    st.metric("Proportion en mode majeur",results['Major Mode Proportion'])
    if results['Any Instrumental Tracks']:
            st.metric("Pistes instrumentales", "Oui")
    else:
            st.metric("Pistes instrumentales", "Non")
            
    with st.expander("Voir plus de statistiques"):
        st.write(f"Track avec la plus grande vivacité:",results['Track with Highest Liveness'])
        st.write(f"Variance en danceabilité:",results['Variance Danceability'])

        
    

def main():
    # Set up Spotify API credentials
    client_id = '23c4e2e1e192402d82e775935d44042f'
    client_secret = 'c864cd4357e7421aae7ff19ac6b6fcea'
    client_credentials_manager = SpotifyClientCredentials(client_id,client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    st.sidebar.title('Navigation')
    choice = st.sidebar.radio('Choisir une page:', ['Accueil', 'Statistiques', 'Visualisations'])
    try:
        mean_popularity = st.session_state['data']['popularity'].mean()
        median_duration = st.session_state['data']['duration_ms'].median()
    except:
        mean_popularity = 0
        median_duration = 0
    try:
        # 3. Mode of time signature
        mode_time_signature = st.session_state['data']['time_signature'].mode()[0]
    except:
        mode_time_signature = 0
    try:
        range_loudness = st.session_state['data']['loudness'].max() - st.session_state['data']['loudness'].min()
    except:
        range_loudness = 0
    try:
        # 5. Standard deviation of tempo
        std_tempo = st.session_state['data']['tempo'].std()
    except:
        std_tempo = 0
    try:
        # 6. Correlation between danceability and energy
        correlation_dance_energy = st.session_state['data']['danceability'].corr(st.session_state['data']['energy'])
    except:
        correlation_dance_energy = 0
    try:
        # 7. Regression line for valence and popularity
        slope, intercept, r_value, p_value, std_err = stats.linregress(st.session_state['data']['valence'], st.session_state['data']['popularity'])
        regression_line = (slope, intercept)
    except:
        regression_line = 0
    try:
        # 8. Mean difference in days between snapshot date and album release date
        st.session_state['data']['days_between'] = (st.session_state['data']['snapshot_date'] - st.session_state['data']['album_release_date']).dt.days
        mean_days_between = st.session_state['data']['days_between'].mean()
    except:
        mean_days_between = 0
    try:
        # 9. Check if explicit content has higher popularity
        explicit_popularity = st.session_state['data'][st.session_state['data']['is_explicit']]['popularity'].mean()
        non_explicit_popularity = st.session_state['data'][~st.session_state['data']['is_explicit']]['popularity'].mean()
        explicit_more_popular = explicit_popularity > non_explicit_popularity
    except:
        explicit_popularity = 0
        non_explicit_popularity = 0
        explicit_more_popular = 0
    try:
        # 10. Average speechiness for tracks with time signature of 4
        avg_speechiness_time_sig_4 = st.session_state['data'][st.session_state['data']['time_signature'] == 4]['speechiness'].mean()
    except:
        avg_speechiness_time_sig_4 = 0
    try:
        # 11. Percentage of tracks with a danceability score above 0.7
        perc_danceable_tracks = (st.session_state['data']['danceability'] > 0.7).mean() * 100
    except:
        perc_danceable_tracks = 0
    try:
        # 12. Most common key in the dataset
        common_key = st.session_state['data']['key'].mode()[0]
    except:
        common_key = 0
    try:
        # 13. Difference in mean loudness between explicit and non-explicit tracks
        mean_loudness_explicit = st.session_state['data'][st.session_state['data']['is_explicit']]['loudness'].mean()
        mean_loudness_non_explicit = st.session_state['data'][~st.session_state['data']['is_explicit']]['loudness'].mean()
        diff_mean_loudness_explicitness = mean_loudness_explicit - mean_loudness_non_explicit
    except:
        mean_loudness_explicit = 0
        mean_loudness_non_explicit = 0
        diff_mean_loudness_explicitness = 0
    try:
        # 14. Energy levels across different countries
        energy_by_country = st.session_state['data'].groupby('country')['energy'].mean().to_dict()
    except:
        energy_by_country = 0
    try:
        # 15. Predicted popularity score from valence using the regression model
        st.session_state['data']['predicted_popularity'] = intercept + slope * st.session_state['data']['valence']
    except:
        st.write("Problem with graph")
    try:
        # 16. Checking for any instrumental tracks
        instrumental_tracks = st.session_state['data']['instrumentalness'].apply(lambda x: True if x > 0.5 else False).any()
    except:
        instrumental_tracks = 0
    try:
        # 17. Variance in danceability across the dataset
        variance_danceability = st.session_state['data']['danceability'].var()
    except:
        variance_danceability = 0
    try:
        # 18. Find the track with the highest liveness
        track_highest_liveness = st.session_state['data'].loc[st.session_state['data']['liveness'].idxmax(), 'name']
    except:
        track_highest_liveness = 0
    try:
        # 19. Determine if higher valence is associated with shorter duration
        correlation_valence_duration = st.session_state['data']['valence'].corr(st.session_state['data']['duration_ms'])
    except:
        correlation_valence_duration = 0
    try:
        # 20. Proportion of tracks in major mode (mode=1)
        major_mode_proportion = (st.session_state['data']['mode'] == 1).mean()
    except:
        major_mode_proportion = 0
    
    # Collecting the results to output
    results = {
        'Mean Popularity': mean_popularity,
        'Median Duration (ms)': median_duration,
        'Mode Time Signature': mode_time_signature,
        'Range Loudness (dB)': range_loudness,
        'STD Tempo (BPM)': std_tempo,
        'Correlation Danceability and Energy': correlation_dance_energy,
        'Regression Line Valence-Popularity': regression_line,
        'Mean Days between Snapshot and Album Release': mean_days_between,
        'Explicit More Popular': explicit_more_popular,
        'Avg Speechiness Time Signature 4': avg_speechiness_time_sig_4,
        'Percentage Danceable Tracks > 0.7': perc_danceable_tracks,
        'Most Common Key': common_key,
        'Difference Mean Loudness Explicitness': diff_mean_loudness_explicitness,
        'Energy Levels by Country': energy_by_country,
        'Any Instrumental Tracks': instrumental_tracks,
        'Variance Danceability': variance_danceability,
        'Track with Highest Liveness': track_highest_liveness,
        'Major Mode Proportion': major_mode_proportion
    }
    
    
    
    if choice == 'Accueil':
        home(sp)
    elif choice == 'Statistiques':
        statistics(results)
    elif choice == 'Visualisations':
        visualisation(st.session_state['data'])

if __name__ == '__main__':
    geolocator = Nominatim(user_agent="geoapiExercises")
    load_and_train()
    main()



