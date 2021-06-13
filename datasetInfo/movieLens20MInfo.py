import os
import numpy as np
import pandas as pd


def getDatasetInfo(path):
    mnames = ['movieId', 'title', 'genres']
    rnames = ['userId', 'movieId', 'rating', 'timestamp']
    movies = pd.read_csv(os.path.join(path, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(path, 'ratings.csv'))

    print('---------------------------------- movies details --------------------------------------')
    print(movies[0: 5])
    print(movies[len(movies) - 5: len(movies)])
    print(movies[movies.isnull().values == True])
    for fname in mnames:
        print(fname, movies[fname].nunique())

    maxlen = 0
    genres = {}
    for i in movies['genres'].values:
        genresList = i.split('|')
        maxlen = max(maxlen, len(genresList))
        for genre in genresList:
            genres[genre] = 1
    print("max genres length: {}".format(maxlen))
    print("genres: {}".format(genres.keys()))

    print('--------------------------------- ratings details --------------------------------------')
    print(ratings[0: 5])
    print(ratings[len(ratings) - 5: len(ratings)])
    print(ratings[ratings.isnull().values == True])
    for fname in rnames:
        print(fname, ratings[fname].nunique())

    print('---------------------------------- to numpy array --------------------------------------')
    movies = np.array(movies)
    ratings = np.array(ratings)

    print(movies)
    print(movies[0, 0], type(movies[0, 0]))
    print(movies[0, 1], type(movies[0, 1]))
    print(movies[0, 2], type(movies[0, 2]))

    print(ratings)
    print(ratings[0, 0], type(ratings[0, 0]))
    print(ratings[0, 1], type(ratings[0, 1]))
    print(ratings[0, 2], type(ratings[0, 2]))
    print(ratings[0, 3], type(ratings[0, 3]))

    print('--------------------------------- dataset details --------------------------------------')
    print("size of movies: {}".format(len(movies)))
    print("size of ratings: {}".format(len(ratings)))
    print("average number of movies rated: {}".format(len(ratings) / ratings[ratings.shape[0] - 1, 0]))

    idx = 1
    missing = []
    for i in range(movies.shape[0]):
        if movies[i, 0] != idx:
            while idx != movies[i, 0]:
                missing.append(idx)
                idx = idx + 1
        idx = idx + 1

    print("missing movies: {}".format(len(missing)))
    # print("missing idx: {}".format(missing))

    # for i in missing:
    #    print(np.where(ratings[:, 2] == i))

    print("no missing idx in ratings")


if __name__ == "__main__":
    getDatasetInfo('../dataset/MovieLens20M')
