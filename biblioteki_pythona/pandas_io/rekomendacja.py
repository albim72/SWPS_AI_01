def get_similar_movies(movie_name, min_ratings=50):
    # wektor ocen dla danego filmu
    movie_ratings = user_movie_matrix[movie_name]
    
    # oblicz korelację Pearsona z innymi filmami
    similar_to_movie = user_movie_matrix.corrwith(movie_ratings)
    
    # stwórz DataFrame i dodaj liczby ocen
    corr_movie = pd.DataFrame(similar_to_movie, columns=['PearsonR'])
    corr_movie.dropna(inplace=True)
    
    # dodaj liczby ocen
    rating_counts = data.groupby('title')['rating'].count()
    corr_movie['RatingCount'] = rating_counts
    
    # filtruj i posortuj
    recommendations = corr_movie[corr_movie['RatingCount'] > min_ratings].sort_values('PearsonR', ascending=False)
    
    return recommendations.head(10)

# Przykład użycia
rekomendacje = get_similar_movies('Matrix')
print(rekomendacje)

# Wizualizacja
sns.scatterplot(data=rekomendacje, x='RatingCount', y='PearsonR')
plt.title("Podobieństwo filmów do 'Matrix'")
plt.xlabel("Liczba ocen")
plt.ylabel("Współczynnik korelacji")
plt.show()
