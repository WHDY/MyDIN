import numpy as np
import pandas as pd


def getDatasetInfo(path):
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(path+'/users.dat', sep='::', header=None, names=unames, engine='python')
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(path+'/ratings.dat', sep='::', header=None, names=rnames, engine='python')
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(path+'/movies.dat', sep='::', header=None, names=mnames, engine='python')

    print('---------------------------------- users details ---------------------------------------')
    print(type(users))
    print(users[0: 5])
    print(type(users[0: 5]))
    print(users[len(users) - 5: len(users)])
    print(users[users.isnull().values==True])
    for fname in unames:
        print(fname, users[fname].nunique())

    print('---------------------------------- movies details --------------------------------------')
    print(movies[0: 5])
    print(movies[len(movies) - 5: len(movies)])
    print(movies[movies.isnull().values==True])
    for fname in mnames:
        print(fname, movies[fname].nunique())

    genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western']
    maxlen = 0;
    for i in movies['genres'].values:
        maxlen = max(maxlen, len(i.split('|')))
    print("max genres: {}".format(maxlen))

    print('--------------------------------- ratings details --------------------------------------')
    print(ratings[0: 5])
    print(ratings[len(ratings) - 5: len(ratings)])
    print(ratings[ratings.isnull().values==True])
    for fname in rnames:
        print(fname, ratings[fname].nunique())

    print('---------------------------------- to numpy array --------------------------------------')
    users = np.array(users)
    movies = np.array(movies)
    ratings = np.array(ratings)

    print(users)
    print(users[0, 0], type(users[0, 0]))
    print(users[0, 1], type(users[0, 1]))
    print(users[0, 2], type(users[0, 2]))
    print(users[0, 3], type(users[0, 3]))
    print(users[0, 4], type(users[0, 4]))

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
    print("size of users: {}".format(len(users)))
    print("size of movies: {}".format(len(movies)))
    print("size of ratings: {}".format(len(ratings)))
    print("average number of movies rated: {}".format(len(ratings) / len(users)))

    idx = 1
    missing = []
    for i in range(movies.shape[0]):
        if movies[i, 0] != idx:
            while idx != movies[i, 0]:
                missing.append(idx)
                idx = idx + 1
        idx = idx + 1

    print("missing movies: {}".format(len(missing)))
    print("missing idx: {}".format(missing))

    for i in missing:
       print(np.where(ratings[:, 2] == i))

    print("no missing idx in ratings")


if __name__ == "__main__":
    getDatasetInfo('../dataset/MovieLens1M')
