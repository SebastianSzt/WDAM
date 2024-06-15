import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Wczytanie danych z plików CSV ze strony: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data
# movies = pd.read_csv("RecommendationSystem/Collaborative/movies.csv", usecols=["movieId", "title"])
# ratings = pd.read_csv("RecommendationSystem/Collaborative/ratings.csv", usecols=["userId", "movieId", "rating"])

movies = pd.read_csv("RecommendationSystem/Collaborative/movies-small.csv", usecols=["movieId", "title"])
ratings = pd.read_csv("RecommendationSystem/Collaborative/ratings-small.csv", usecols=["userId", "movieId", "rating"])

# Zbieranie danych od użytkownika
print("Filmy są dostępne w bazie danych. Wybierz co najmniej 5 filmów, które oglądałeś.")

user_ratings = {}

while True:
    movie = input("Podaj tytuł filmu: ")
    matching_movies = movies[movies['title'].str.lower() == movie.lower()]

    if not matching_movies.empty:
        movie_title = matching_movies.iloc[0]['title']
        if movie_title not in user_ratings:
            while True:
                rating = float(input("Podaj ocenę (0.5-5.0): "))
                if 0.5 <= rating <= 5:
                    break
                else:
                    print("Ocena musi być w przedziale 0.5-5.0.")

            user_ratings[movie_title] = rating
        else:
            print("Film już został podany. Wybierz inny film.")
    else:
        print("Tytuł filmu nie znaleziony w bazie. Spróbuj ponownie.")

    if len(user_ratings) >= 5:
        cont = input("Czy chcesz dodać kolejny film? (tak/nie): ").strip().lower()
        if cont == 'nie':
            break

user_ratings_test = {
    "Toy Story (1995)": 4.0,
    "Grumpier Old Men (1995)": 4.0,
    "Heat (1995)": 4.0,
    "Seven (a.k.a. Se7en) (1995)": 5.0,
    "Usual Suspects, The (1995)" : 5.0
}

num_recommendations = int(input("Ile rekomendacji chcesz otrzymać? (podaj liczbę): "))

# Funkcja rekomendująca filmy
def recommend_movies(user_ratings, num_recommendations, calculate_rmse=False):
    # Przypisanie nowego ID użytkownika
    user_id = ratings['userId'].max() + 1
    new_ratings = ratings.copy()
    
    # Dodanie ocen użytkownika do zestawu danych
    for movie, rating in user_ratings.items():
        # Pobranie ID filmu na podstawie jego tytułu
        movie_id = movies[movies['title'] == movie]['movieId'].values[0]
        # Utworzenie nowego wiersza z oceną użytkownika
        new_row = pd.DataFrame({'userId': [user_id], 'movieId': [movie_id], 'rating': [rating]})
        # Dodanie nowego wiersza do zbioru ocen
        new_ratings = pd.concat([new_ratings, new_row], ignore_index=True)
    
    # Przygotowanie danych do biblioteki surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(new_ratings, reader)
   
    if calculate_rmse:
        # Podział danych na zestaw treningowy i testowy
        trainset, testset = train_test_split(data, test_size=0.2)
        # Trenowanie modelu SVD na nowym zbiorze danych
        svd = SVD()
        svd.fit(trainset)
        # Kalkulacja RMSE
        predictions = svd.test(testset)
        rmse = accuracy.rmse(predictions)
    else:
        # Budowanie pełnego zbioru treningowego z nowych danych
        trainset = data.build_full_trainset()
        # Trenowanie modelu SVD na nowym zbiorze danych
        svd = SVD()
        svd.fit(trainset)
    
    # Pobranie unikalnych ID filmów
    movie_ids = movies['movieId'].unique()
    # Lista obejrzanych przez użytkownika filmów na podstawie wprowadzonych ocen
    watched_movie_ids = [movies[movies['title'] == movie]['movieId'].values[0] for movie in user_ratings]
    recommendations = []

    # Przewidywanie ocen dla filmów, których użytkownik jeszcze nie widział
    for movie_id in movie_ids:
        if movie_id not in watched_movie_ids:
            # Przewidywanie oceny filmu dla nowego użytkownika
            est = svd.predict(user_id, movie_id).est
            recommendations.append((movie_id, est))
    
    # Sortowanie rekomendacji według przewidywanej oceny w porządku malejącym
    recommendations.sort(key=lambda x: x[1], reverse=True)
    # Wybieranie ID najlepszych rekomendacji
    recommended_movie_ids = [rec[0] for rec in recommendations[:num_recommendations]]
    # Pobranie tytułów rekomendowanych filmów na podstawie ich ID
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    
    return recommended_movies

recommended_movies = recommend_movies(user_ratings, num_recommendations)
print("\nRekomendowane filmy:")
# print(recommended_movies, "\n") # Wyświetlenie pełnych danych
print("\n".join(recommended_movies['title']), "\n")