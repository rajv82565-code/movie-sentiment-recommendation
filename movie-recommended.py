import pandas as pd
import numpy as np
import os

def load_movie_data():
    movies_path = 'movies.csv/movies.csv' if os.path.exists('movies.csv/movies.csv') else 'movies.csv'
    movies_df = pd.read_csv(movies_path)
    
    # Preprocessing
    movies_df['year'] = movies_df.title.str.extract(r'(\(\d\d\d\d\))', expand=False)
    movies_df['year'] = movies_df.year.str.extract(r'(\d\d\d\d)', expand=False)
    movies_df['title_clean'] = movies_df['title'].str.replace(r'\(\d{4}\)', '', regex=True).apply(lambda x: x.strip())
    
    # Process genres
    movies_df['genres_list'] = movies_df.genres.str.split('|')
    
    all_genres = set()
    for genres in movies_df['genres_list']:
        all_genres.update(genres)
    
    moviesWithGenres_df = movies_df.copy()
    for genre in all_genres:
        moviesWithGenres_df[genre] = moviesWithGenres_df['genres_list'].apply(lambda x: 1 if genre in x else 0)
        
    return movies_df, moviesWithGenres_df, all_genres

def get_recommendations(movie_title, movies_df, moviesWithGenres_df, all_genres):
    movie_row = movies_df[movies_df['title_clean'].str.lower() == movie_title.lower()]
    if movie_row.empty:
        print(f"Movie '{movie_title}' not found in database.")
        return None
    
    movie_id = movie_row.iloc[0]['movieId']
    selected_movie_genres = moviesWithGenres_df[moviesWithGenres_df['movieId'] == movie_id][list(all_genres)].iloc[0]
    
    genre_matrix = moviesWithGenres_df[list(all_genres)]
    scores = genre_matrix.dot(selected_movie_genres)
    
    recommendations_idx = scores.sort_values(ascending=False).head(6).index
    recommendations = movies_df.iloc[recommendations_idx]
    recommendations = recommendations[recommendations['movieId'] != movie_id].head(5)
    return recommendations

if __name__ == "__main__":
    movies_df, moviesWithGenres_df, all_genres = load_movie_data()
    movie_query = "Toy Story"
    print(f"Recommendations for {movie_query}:")
    recs = get_recommendations(movie_query, movies_df, moviesWithGenres_df, all_genres)
    if recs is not None:
        print(recs[['title_clean', 'year']])
