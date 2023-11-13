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


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'



#%% MISE EN PLACE DE LA SESSION STREAMLIT POUR POUVOIR CHARGER ET ENTRAINER LE MODEL

@st.cache_data
def load_and_train():
    if 'data' not in st.session_state or 'model' not in st.session_state:
        #%% TRAITEMENT DE LA DONNEE
        columns_to_load = ["name","artists","daily_rank","daily_movement","weekly_movement","country","snapshot_date","popularity","is_explicit","duration_ms","album_name","album_release_date","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","time_signature"] 
        st.session_state['data'] = pd.read_csv("C://Users//henri//Downloads//archive (4)//universal_top_spotify_songs.csv", usecols=columns_to_load,sep=",")
        st.session_state['features'] = st.session_state['data'][['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'mode']]
        st.session_state['target'] = np.random.choice(['Rock', 'Pop', 'Jazz', 'Hip-Hop'], size=len(st.session_state['data']))
        
        for col in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']:
            st.session_state['data'][col] = st.session_state['data'][col].astype(str).str.replace(',', '.').astype(float)
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



def plot_valence_popularity_regression(data):
    fig = px.scatter(data, x='valence', y='popularity', trendline='ols', title='Valence et Popularité')
    st.plotly_chart(fig)

def plot_time_signature_distribution(data):
    fig = px.bar(data['time_signature'].value_counts().reset_index(), x='index', y='time_signature', title='Répartition des Signatures Temporelles')
    st.plotly_chart(fig)


def plot_explicit_proportion(data):
    fig = px.pie(data, names='is_explicit', title='Proportion de Morceaux Explicites')
    st.plotly_chart(fig)


def plot_common_key(data):
    fig = px.bar(data['key'].value_counts().reset_index(), x='index', y='key', title='Clé Musicale la Plus Commune')
    st.plotly_chart(fig)


def plot_valence_vs_duration(data):
    fig = px.scatter(data, x='valence', y='duration_ms', title='Valence contre Durée')
    st.plotly_chart(fig)

def plot_liveness_distribution(data):
    fig = px.histogram(data, x='liveness', title='Distribution de la Liveness')
    st.plotly_chart(fig)


# Note: Ceci nécessite que la colonne 'genre' soit présente dans vos données
def plot_genre_distribution(data):
    fig = px.bar(data['genre'].value_counts().reset_index(), x='index', y='genre', title='Genres Musicaux les Plus Communs')
    st.plotly_chart(fig)


def plot_mode_distribution(data):
    fig = px.pie(data, names='mode', title='Répartition des Modes (Majeur/Mineur)')
    st.plotly_chart(fig)


    
def plot_feature_correlation(data):
    corr = data[['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title='Corrélation entre Caractéristiques Musicales')
    st.plotly_chart(fig)


def plot_loudness_boxplot(data):
    fig = px.box(data, y='loudness', title='Niveaux de Loudness')
    st.plotly_chart(fig)

    
    
def plot_popularity_distribution(data):
    # Use a sidebar slider to select the number of bins
    bins = st.sidebar.slider('Select number of bins for popularity distribution', 5, 50, 20)
    fig = px.histogram(data, x='popularity', nbins=bins, title='Distribution de la Popularité')
    st.plotly_chart(fig)


# Energy by Country with Sorting Selector
def plot_energy_by_country(data):
    # Use a sidebar radio button to select the sort order
    sort_order = st.sidebar.radio('Select sort order for energy by country', ['Ascending', 'Descending'])
    energy_country = data.groupby('country')['energy'].mean().reset_index()
    energy_country = energy_country.sort_values(by='energy', ascending=(sort_order == 'Ascending'))
    fig = px.bar(energy_country, y='country', x='energy', orientation='h', title='Energie Moyenne par Pays')
    st.plotly_chart(fig)

# Danceability vs Energy with Alpha Selector
def plot_danceability_vs_energy(data):
    alpha = st.sidebar.slider('Select alpha for danceability vs energy plot', 
                              min_value=0.1, max_value=1.0, value=0.5, key='alpha_slider')
    fig = px.scatter(data, x='danceability', y='energy', opacity=alpha,
                     title='Danceabilité contre Energie', labels={'danceability': 'Danceabilité', 'energy': 'Energie'})
    st.plotly_chart(fig)

def plot_duration_distribution(data):
    unit = st.sidebar.radio('Select duration unit', ('Milliseconds', 'Minutes'), key='duration_unit')
    bins = st.sidebar.slider('Select number of bins for duration distribution', min_value=5, max_value=50, value=20, key='duration_bins')
    durations = data['duration_ms'] if unit == 'Milliseconds' else data['duration_ms'] / 60000
    title = 'Distribution des Durées des Morceaux' if unit == 'Milliseconds' else 'Distribution des Durées des Morceaux (en Minutes)'
    fig = px.histogram(data, x=durations, nbins=bins, title=title)
    fig.update_layout(xaxis_title=f'Durée ({unit})', yaxis_title='Nombre de Morceaux')
    st.plotly_chart(fig)


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

    folium_static(m)
    plot_popularity_distribution(data)
    plot_energy_by_country(data)
    plot_danceability_vs_energy(data)
    plot_duration_distribution(data)
    plot_loudness_boxplot(data)
    plot_feature_correlation(data)
    plot_mode_distribution(data)
    plot_genre_distribution(data)
    plot_liveness_distribution(data)
    plot_valence_vs_duration(data)
    plot_common_key(data)
    plot_explicit_proportion(data)
    plot_time_signature_distribution(data)
    plot_valence_popularity_regression(data)
    data['popularity_moving_average'] = data['popularity'].rolling(window=7).mean()

    # Create a Bokeh chart (e.g., popularity trend)
    p = figure(title='Popularity Trend', x_axis_label='Date', y_axis_label='Popularity Moving Average', x_axis_type='datetime')
    p.line(data['snapshot_date'], data['popularity_moving_average'], legend_label='Popularity Trend', line_width=2)
    p.add_tools(HoverTool(tooltips=[('Date', '@x{%F}'), ('Popularity', '@y')], formatters={'@x': 'datetime'}))

    # Display the Bokeh chart in Streamlit
    st.bokeh_chart(p, use_container_width=True)

    

def home():
    st.title('Spotify Data Analysis')
    
    # Using column layout for better control
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image('starry-sky-8199764_1280.jpg', width=200)
    
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
    st.subheader('Meet the Team')
    st.markdown('''
    - **Henri Serano**: Le big boss
    - **Sara Thibierge**: La putchiste
    - **Eloi Seidlitz**: Le mec du Nord
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
    st.sidebar.title('Navigation')
    choice = st.sidebar.radio('Choisir une page:', ['Accueil', 'Statistiques', 'Visualisations'])
    mean_popularity = st.session_state['data']['popularity'].mean()
    median_duration = st.session_state['data']['duration_ms'].median()
    
    # 3. Mode of time signature
    mode_time_signature = st.session_state['data']['time_signature'].mode()[0]
    range_loudness = st.session_state['data']['loudness'].max() - st.session_state['data']['loudness'].min()

    # 5. Standard deviation of tempo
    std_tempo = st.session_state['data']['tempo'].std()

    # 6. Correlation between danceability and energy
    correlation_dance_energy = st.session_state['data']['danceability'].corr(st.session_state['data']['energy'])

    # 7. Regression line for valence and popularity
    slope, intercept, r_value, p_value, std_err = stats.linregress(st.session_state['data']['valence'], st.session_state['data']['popularity'])
    regression_line = (slope, intercept)

    # 8. Mean difference in days between snapshot date and album release date
    st.session_state['data']['days_between'] = (st.session_state['data']['snapshot_date'] - st.session_state['data']['album_release_date']).dt.days
    mean_days_between = st.session_state['data']['days_between'].mean()

    # 9. Check if explicit content has higher popularity
    explicit_popularity = st.session_state['data'][st.session_state['data']['is_explicit']]['popularity'].mean()
    non_explicit_popularity = st.session_state['data'][~st.session_state['data']['is_explicit']]['popularity'].mean()
    explicit_more_popular = explicit_popularity > non_explicit_popularity

    # 10. Average speechiness for tracks with time signature of 4
    avg_speechiness_time_sig_4 = st.session_state['data'][st.session_state['data']['time_signature'] == 4]['speechiness'].mean()

    # 11. Percentage of tracks with a danceability score above 0.7
    perc_danceable_tracks = (st.session_state['data']['danceability'] > 0.7).mean() * 100

    # 12. Most common key in the dataset
    common_key = st.session_state['data']['key'].mode()[0]

    # 13. Difference in mean loudness between explicit and non-explicit tracks
    mean_loudness_explicit = st.session_state['data'][st.session_state['data']['is_explicit']]['loudness'].mean()
    mean_loudness_non_explicit = st.session_state['data'][~st.session_state['data']['is_explicit']]['loudness'].mean()
    diff_mean_loudness_explicitness = mean_loudness_explicit - mean_loudness_non_explicit

    # 14. Energy levels across different countries
    energy_by_country = st.session_state['data'].groupby('country')['energy'].mean().to_dict()

    # 15. Predicted popularity score from valence using the regression model
    st.session_state['data']['predicted_popularity'] = intercept + slope * st.session_state['data']['valence']

    # 16. Checking for any instrumental tracks
    instrumental_tracks = st.session_state['data']['instrumentalness'].apply(lambda x: True if x > 0.5 else False).any()

    # 17. Variance in danceability across the dataset
    variance_danceability = st.session_state['data']['danceability'].var()

    # 18. Find the track with the highest liveness
    track_highest_liveness = st.session_state['data'].loc[st.session_state['data']['liveness'].idxmax(), 'name']

    # 19. Determine if higher valence is associated with shorter duration
    correlation_valence_duration = st.session_state['data']['valence'].corr(st.session_state['data']['duration_ms'])

    # 20. Proportion of tracks in major mode (mode=1)
    major_mode_proportion = (st.session_state['data']['mode'] == 1).mean()

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
        home()
    elif choice == 'Statistiques':
        statistics(results)
    elif choice == 'Visualisations':
        visualisation(st.session_state['data'])

if __name__ == '__main__':
    geolocator = Nominatim(user_agent="geoapiExercises")
    load_and_train()
    main()



