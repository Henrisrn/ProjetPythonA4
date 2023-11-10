# %%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


data = pd.read_csv("C://Users//henri//Downloads//archive (4)//universal_top_spotify_songs.csv",sep=",")

# %%
for col in ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']:
    data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# Converting to datetime
data['snapshot_date'] = pd.to_datetime(data['snapshot_date'])
data['album_release_date'] = pd.to_datetime(data['album_release_date'])

# Replace the country codes with full names for clarity
country_codes = {'SK': 'South Korea', 'CA': 'Canada', 'ES': 'Spain', 'PR': 'Puerto Rico'}
data['country'] = data['country'].map(country_codes)

# Let's start with some basic statistics and inferences.

# 1. Mean popularity
mean_popularity = data['popularity'].mean()

# 2. Median duration in ms
median_duration = data['duration_ms'].median()

# 3. Mode of time signature
mode_time_signature = data['time_signature'].mode()[0]


# %%
range_loudness = data['loudness'].max() - data['loudness'].min()

# 5. Standard deviation of tempo
std_tempo = data['tempo'].std()

# 6. Correlation between danceability and energy
correlation_dance_energy = data['danceability'].corr(data['energy'])

# 7. Regression line for valence and popularity
slope, intercept, r_value, p_value, std_err = stats.linregress(data['valence'], data['popularity'])
regression_line = (slope, intercept)

# 8. Mean difference in days between snapshot date and album release date
data['days_between'] = (data['snapshot_date'] - data['album_release_date']).dt.days
mean_days_between = data['days_between'].mean()

# 9. Check if explicit content has higher popularity
explicit_popularity = data[data['is_explicit']]['popularity'].mean()
non_explicit_popularity = data[~data['is_explicit']]['popularity'].mean()
explicit_more_popular = explicit_popularity > non_explicit_popularity

# 10. Average speechiness for tracks with time signature of 4
avg_speechiness_time_sig_4 = data[data['time_signature'] == 4]['speechiness'].mean()

# 11. Percentage of tracks with a danceability score above 0.7
perc_danceable_tracks = (data['danceability'] > 0.7).mean() * 100

# 12. Most common key in the dataset
common_key = data['key'].mode()[0]

# 13. Difference in mean loudness between explicit and non-explicit tracks
mean_loudness_explicit = data[data['is_explicit']]['loudness'].mean()
mean_loudness_non_explicit = data[~data['is_explicit']]['loudness'].mean()
diff_mean_loudness_explicitness = mean_loudness_explicit - mean_loudness_non_explicit

# 14. Energy levels across different countries
energy_by_country = data.groupby('country')['energy'].mean().to_dict()

# 15. Predicted popularity score from valence using the regression model
data['predicted_popularity'] = intercept + slope * data['valence']

# 16. Checking for any instrumental tracks
instrumental_tracks = data['instrumentalness'].apply(lambda x: True if x > 0.5 else False).any()

# 17. Variance in danceability across the dataset
variance_danceability = data['danceability'].var()

# 18. Find the track with the highest liveness
track_highest_liveness = data.loc[data['liveness'].idxmax(), 'name']

# 19. Determine if higher valence is associated with shorter duration
correlation_valence_duration = data['valence'].corr(data['duration_ms'])

# 20. Proportion of tracks in major mode (mode=1)
major_mode_proportion = (data['mode'] == 1).mean()

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
    'Predicted Popularity from Valence': data['predicted_popularity'].tolist(),
    'Any Instrumental Tracks': instrumental_tracks,
    'Variance Danceability': variance_danceability,
    'Track with Highest Liveness': track_highest_liveness,
    'Correlation Valence and Duration': correlation_valence_duration,
    'Major Mode Proportion': major_mode_proportion
}
features = data[['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'mode']]

# Target labels - This should be your genre column which needs to be encoded if it's categorical
# For the purpose of this example, I'm going to create a random target array
target = np.random.choice(['Rock', 'Pop', 'Jazz', 'Hip-Hop'], size=len(data))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
full_data_predictions = clf.predict(scaler.transform(features))

# Add the predictions to the DataFrame
data['genre'] = full_data_predictions
# %%
def plot_popularity_distribution(data):
    plt.hist(data['popularity'], bins=20, color='blue')
    plt.title('Distribution de la Popularité')
    plt.xlabel('Popularité')
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)
def plot_energy_by_country(data):
    energy_country = data.groupby('country')['energy'].mean().sort_values()
    energy_country.plot(kind='barh', color='green')
    plt.title('Energie Moyenne par Pays')
    plt.xlabel('Energie Moyenne')
    plt.ylabel('Pays')
    st.pyplot(plt)

def plot_danceability_vs_energy(data):
    plt.scatter(data['danceability'], data['energy'], alpha=0.5)
    plt.title('Danceabilité contre Energie')
    plt.xlabel('Danceabilité')
    plt.ylabel('Energie')
    st.pyplot(plt)

def plot_duration_distribution(data):
    plt.hist(data['duration_ms']/60000, bins=20, color='purple')  # Convert ms to minutes
    plt.title('Distribution des Durées des Morceaux')
    plt.xlabel('Durée (Minutes)')
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)

def plot_valence_popularity_regression(data):
    slope, intercept = np.polyfit(data['valence'], data['popularity'], 1)
    plt.scatter(data['valence'], data['popularity'], alpha=0.5)
    plt.plot(data['valence'], intercept + slope * data['valence'], color='red')
    plt.title('Valence et Popularité')
    plt.xlabel('Valence')
    plt.ylabel('Popularité')
    st.pyplot(plt)

def plot_time_signature_distribution(data):
    data['time_signature'].value_counts().plot(kind='bar', color='orange')
    plt.title('Répartition des Signatures Temporelles')
    plt.xlabel('Signature Temporelle')
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)

def plot_explicit_proportion(data):
    data['is_explicit'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen'])
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Proportion de Morceaux Explicites')
    st.pyplot(plt)

def plot_common_key(data):
    data['key'].value_counts().plot(kind='bar', color='teal')
    plt.title('Clé Musicale la Plus Commune')
    plt.xlabel('Clé')
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)

def plot_valence_vs_duration(data):
    plt.scatter(data['valence'], data['duration_ms']/60000, alpha=0.5)  # Convert ms to minutes
    plt.title('Valence contre Durée')
    plt.xlabel('Valence')
    plt.ylabel('Durée (Minutes)')
    st.pyplot(plt)

def plot_liveness_distribution(data):
    plt.hist(data['liveness'], bins=20, color='magenta')
    plt.title('Distribution de la Liveness')
    plt.xlabel('Liveness')
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)

# Note: Ceci nécessite que la colonne 'genre' soit présente dans vos données
def plot_genre_distribution(data):
    data['genre'].value_counts().head(10).plot(kind='bar', color='brown')
    plt.title('Genres Musicaux les Plus Communs')
    plt.xlabel('Genre')
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)

def plot_mode_distribution(data):
    data['mode'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    plt.axis('equal')
    plt.title('Répartition des Modes (Majeur/Mineur)')
    st.pyplot(plt)

def plot_feature_correlation(data):
    corr = data[['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Corrélation entre Caractéristiques Musicales')
    st.pyplot(plt)

def plot_loudness_boxplot(data):
    plt.boxplot(data['loudness'])
    plt.title('Niveaux de Loudness')
    plt.ylabel('Loudness (dB)')
    st.pyplot(plt)
def plot_popularity_distribution(data):
    bins = st.sidebar.slider('Select number of bins for popularity distribution', min_value=5, max_value=50, value=20)
    plt.hist(data['popularity'], bins=bins, color='blue')
    plt.title('Distribution de la Popularité')
    plt.xlabel('Popularité')
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)

# Energy by Country with Sorting Selector
def plot_energy_by_country(data):
    sort_order = st.sidebar.radio('Select sort order for energy by country', ('Ascending', 'Descending'))
    energy_country = data.groupby('country')['energy'].mean().sort_values(ascending=(sort_order == 'Ascending'))
    energy_country.plot(kind='barh', color='green')
    plt.title('Energie Moyenne par Pays')
    plt.xlabel('Energie Moyenne')
    plt.ylabel('Pays')
    st.pyplot(plt)

# Danceability vs Energy with Alpha Selector
def plot_danceability_vs_energy(data):
    alpha = st.sidebar.slider('Select alpha for danceability vs energy plot', 
                          min_value=0.1, max_value=1.0, value=0.5, key='alpha_slider')

    plt.scatter(data['danceability'], data['energy'], alpha=alpha)
    plt.title('Danceabilité contre Energie')
    plt.xlabel('Danceabilité')
    plt.ylabel('Energie')
    st.pyplot(plt)

# Duration Distribution with Conversion and Bin Selector
def plot_duration_distribution(data):
    unit = st.sidebar.radio('Select duration unit', ('Milliseconds', 'Minutes'))
    bins = st.sidebar.slider('Select number of bins for duration distribution', min_value=5, max_value=50, value=20)
    durations = data['duration_ms'] if unit == 'Milliseconds' else data['duration_ms'] / 60000
    plt.hist(durations, bins=bins, color='purple')
    plt.title('Distribution des Durées des Morceaux')
    plt.xlabel('Durée ({})'.format(unit))
    plt.ylabel('Nombre de Morceaux')
    st.pyplot(plt)
def home():
    st.title('Analyse Spotify')
    st.write('Bienvenue dans l\'application d\'analyse des données Spotify.')
    plot_popularity_distribution(data)
    plot_energy_by_country(data)
    plot_danceability_vs_energy(data)
    plot_duration_distribution(data)
    plot_loudness_boxplot(data)
    plot_feature_correlation(data)
    plot_mode_distribution(data)
# Fonction pour afficher les statistiques
def statistics():
    st.write('Statistiques de base :')
    st.write('Moyenne de popularité :', mean_popularity)
    # Continuer pour les autres statistiques...

def main():
    st.sidebar.title('Navigation')
    choice = st.sidebar.radio('Choisir une page:', ['Accueil', 'Statistiques', 'Visualisations'])
    plot_choice = st.sidebar.selectbox('Choose the plot', [
    'Popularity Distribution',
    'Energy by Country',
    'Danceability vs Energy',
    'Duration Distribution',
    'Valence and Popularity Regression',
    'Time Signature Distribution',
    'Explicit Content Proportion',
    'Common Key Distribution',
    'Valence vs Duration',
    'Liveness Distribution',
    'Genre Distribution',  # Assuming 'genre' column exists
    'Mode Distribution',
    'Feature Correlation',
    'Loudness Boxplot',
    ])

    # Display the chosen plot
    if plot_choice == 'Popularity Distribution':
        plot_popularity_distribution(data)
    elif plot_choice == 'Energy by Country':
        plot_energy_by_country(data)
    elif plot_choice == 'Danceability vs Energy':
        plot_danceability_vs_energy(data)
    elif plot_choice == 'Duration Distribution':
        plot_duration_distribution(data)
    elif plot_choice == 'Valence and Popularity Regression':
        plot_valence_popularity_regression(data)
    elif plot_choice == 'Time Signature Distribution':
        plot_time_signature_distribution(data)
    elif plot_choice == 'Explicit Content Proportion':
        plot_explicit_proportion(data)
    elif plot_choice == 'Common Key Distribution':
        plot_common_key(data)
    elif plot_choice == 'Valence vs Duration':
        plot_valence_vs_duration(data)
    elif plot_choice == 'Liveness Distribution':
        plot_liveness_distribution(data)
    elif plot_choice == 'Genre Distribution':
        plot_genre_distribution(data)  # This needs the 'genre' column in your dataset
    elif plot_choice == 'Mode Distribution':
        plot_mode_distribution(data)
    elif plot_choice == 'Feature Correlation':
        plot_feature_correlation(data)
    elif plot_choice == 'Loudness Boxplot':
        plot_loudness_boxplot(data)

    if choice == 'Accueil':
        home()
    elif choice == 'Statistiques':
        statistics()
# %%
if __name__ == '__main__':
    main()

# %%
"""print(data.columns)
data.to_excel("DatasetSpotify.xlsx",index=False)"""


