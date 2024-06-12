import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re


# data = pd.read_csv('RecommendationSystem/imdb.csv')
# data[['Title', 'Genre', 'Description', 'Director', 'Actors']] = data[['Title', 'Genre', 'Description', 'Director', 'Actors']].fillna('')
# data[['Rank', 'Year', 'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)', 'Metascore']] = data[['Rank', 'Year', 'Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)', 'Metascore']].fillna(0)
# data['features'] = (data['Title'] + ' ' + data['Genre'] + ' ' + data['Description'] + ' ' + data['Director'] + ' ' + data['Actors'])
# num_features = data[['Year', 'Rating', 'Revenue (Millions)', 'Metascore']].values


# Dane ze strony https://www.kaggle.com/datasets/danielgrijalvas/movies
# Wczytanie danych z pliku CSV
data = pd.read_csv('RecommendationSystem/movies.csv')

# Usunięcie informacji o kraju z kolumny 'year'
data['year'] = data['year'].apply(lambda x: re.sub(r' \((.*?)\)', '', str(x)))

# Wypełnienie brakujących wartości pustym ciągiem dla kolumn tekstowych
data[['name', 'rating', 'genre', 'director', 'writer', 'star', 'country', 'company']] = data[['name', 'rating', 'genre', 'director', 'writer', 'star', 'country', 'company']].fillna('')

# Wypełnienie brakujących wartości zerami dla kolumn numerycznych
data[['year', 'score', 'votes', 'budget', 'gross', 'runtime']] = data[['year', 'score', 'votes', 'budget', 'gross', 'runtime']].fillna(0)

# Połączenie wszystkich istotnych kolumn tekstowych w jedną kolumnę 'features'
data['features'] = (data['name'] + ' ' + data['rating'] + ' ' + data['genre'] + ' ' + data['director'] + ' ' + data['writer'] + ' ' + data['star'] + ' ' + data['country'] + ' ' + data['company'])

# Przekształcenie tekstu na wektory cech przy użyciu TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(data['features'])

# Przekształcenie numerycznych kolumn na macierz
num_features = data[['year', 'score', 'votes', 'budget', 'gross']].values

# Normalizacja numerycznych cech
scaler = MinMaxScaler()
num_features_scaled = scaler.fit_transform(num_features)

# Połączenie wektorów cech tekstowych i numerycznych
features = np.hstack((matrix.toarray(), num_features_scaled))

# Obliczenie macierzy podobieństwa cosinusowego
cosine_sim = cosine_similarity(features, features)

# Funkcja rekomendująca filmy na podstawie wybranych filmów
def recommend_movies(selected_movies, num_recommendations, cosine_sim=cosine_sim, data=data):
    # Utworzenie serii z indeksami filmów na podstawie tytułu i roku
    indices = pd.Series(data.index, index=data[['name', 'year']].apply(lambda x: f"{x['name']} ({x['year']})", axis=1)).drop_duplicates()
    # Znalezienie indeksów wybranych filmów
    selected_indices = [indices[f"{title} ({year})"] for title, year in selected_movies]
    
    '''
    # Sprawdzenie kształtu wybranych wektorów cech
    for idx in selected_indices:
        print(f"Index: {idx}, Feature vector shape: {cosine_sim[idx].shape}")
    '''
    
    # Obliczenie średniego podobieństwa do wybranych filmów
    sim_scores = np.mean(cosine_sim[selected_indices], axis=0)
    
    # Posortowanie filmów według podobieństwa
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Pobranie indeksów najbardziej podobnych filmów, pomijając wybrane filmy
    movie_indices = [i[0] for i in sim_scores if i[0] not in selected_indices]
    recommended_movies = data['name'].iloc[movie_indices].head(num_recommendations)
    
    return recommended_movies

print("Filmy są dostępne w bazie danych. Wybierz co najmniej 5 filmów, które oglądałeś.")

selected_movies = []
while True:
    movie = input("Podaj tytuł filmu, który oglądałeś: ")
    matching_movies = data[data['name'].str.lower() == movie.lower()]
    
    if not matching_movies.empty:
        if len(matching_movies) == 1:
            selected_movies.append((matching_movies.iloc[0]['name'], matching_movies.iloc[0]['year']))
        else:
            print("Znaleziono kilka filmów o tym tytule:")
            for i, row in matching_movies.iterrows():
                print(f"{row['name']} ({row['year']})")
            year = input("Podaj rok wydania filmu: ")
            matching_movie = matching_movies[matching_movies['year'] == year]
            if not matching_movie.empty:
                selected_movies.append((matching_movie.iloc[0]['name'], matching_movie.iloc[0]['year']))
            else:
                print("Nie znaleziono filmu o takim tytule i roku. Spróbuj ponownie.")
    else:
        print("Tytuł filmu nie znaleziony w bazie. Spróbuj ponownie.")
    
    if len(selected_movies) >= 5:
        cont = input("Czy chcesz dodać kolejny film? (tak/nie): ").strip().lower()
        if cont == 'nie':
            break

num_recommendations = int(input("Ile rekomendacji chcesz otrzymać? (podaj liczbę): "))

recommended_movies = recommend_movies(selected_movies, num_recommendations)
print("\nRekomendowane filmy:")
for movie in recommended_movies:
    print(movie)