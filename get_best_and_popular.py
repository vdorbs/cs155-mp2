#!/usr/bin/env python3
import numpy as np
from basic_visual import parse_ids_by_genre

def get_best_and_popular():
    data = np.loadtxt('data/data.txt', delimiter='\t')
    ids_by_genre = parse_ids_by_genre('data/movies.txt')

    ratings_by_id = {movie_id:[] for _, movie_id, _ in data}
    for _, movie_id, rating in data:
        ratings_by_id[movie_id].append(rating)
    average = lambda r: sum(r) / len(r)
    ratings_most_popular = sorted(ratings_by_id.items(),
                                  key=lambda k: len(k[1]),
                                  reverse=True)[:10]
    ratings_best = sorted(ratings_by_id.items(),
                          key=lambda k: sum(k[1]) / len(k[1]),
                          reverse=True)[:10]
    ratings_most_popular = [int(k) for k, v in ratings_most_popular]
    ratings_best = [int(k) for k, v in ratings_best]
    return ratings_best, ratings_most_popular

if __name__ == '__main__':
    best, most_popular = get_best_and_popular()
    print(best)
    print(most_popular)
