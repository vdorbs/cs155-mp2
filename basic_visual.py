import numpy as np
import matplotlib.pyplot as plt

def parse_ids_by_genre(path):
    GENRE_LIST = [
        'Unknown', 'Action', 'Adventure', 'Animation', 'Childrens',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    ids_by_genre = {genre:[] for genre in GENRE_LIST}
    with open('data/movies.txt', 'r') as fid:
        for line in fid:
            line = line.rstrip('\n').split('\t')
            movie_id = int(line[0])
            for idx, is_genre in enumerate(line[2:]):
                if is_genre == '1':
                    ids_by_genre[GENRE_LIST[idx]].append(movie_id)
    return ids_by_genre

def do_many_things():
    data = np.loadtxt('data/data.txt', delimiter='\t')
    ids_by_genre = parse_ids_by_genre('data/movies.txt')
    ratings_all = data[:, 2]

    best = lambda ratings: sum(ratings) / len(ratings)
    ratings_by_id = {movie_id:[] for _, movie_id, _ in data}
    for _, movie_id, rating in data:
        ratings_by_id[movie_id].append(rating)
    ratings_most_popular = sorted(ratings_by_id.values(), key=len, reverse=True)[:10]
    ratings_best = sorted(ratings_by_id.values(), key=best, reverse=True)[:10]
    ratings_most_popular = [rating for ratings in ratings_most_popular for rating in ratings]
    ratings_best = [rating for ratings in ratings_best for rating in ratings]

    genres = ['Documentary', 'Romance', 'Western']
    ratings_genres = []
    for genre in genres:
        movie_ids = ids_by_genre[genre]
        ratings_genre = []
        for movie_id in movie_ids:
            ratings_genre.extend(ratings_by_id[movie_id])
        ratings_genres.append(ratings_genre)

    rating_sets = [ratings_all, ratings_most_popular, ratings_best] + ratings_genres
    titles = ['All ratings', 'Ratings (most popular)', 'Ratings (best)'] + genres
    for rating_set, title in zip(rating_sets, titles):
        plt.figure()
        plt.hist(rating_set, 5)
        plt.title(title)
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.savefig('figures/' + title)

if __name__ == '__main__':
    do_many_things()
