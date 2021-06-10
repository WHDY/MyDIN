import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Binarizer


class dataGenerator(object):
    """
    preprocess and make dataset
    """
    def __init__(self, dataPath, ratingBinThreshold, maxSequenceLen, splitRatio=0.7, splitMehtod='behavior'):
        super(dataGenerator, self).__init__()

        self.ratingThreshold = ratingBinThreshold
        self.maxSequenceLen = maxSequenceLen
        self.splitRation = splitRatio
        self.splitMethod = splitMehtod

        # load data
        print("preparing data...")
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        mnames = ['movie_id', 'title', 'genres']
        self.userFeatures = pd.read_table(os.path.join(dataPath, 'users.dat'),
                                          sep='::', header=None, names=unames, engine='python')
        self.movieFeatures = pd.read_table(os.path.join(dataPath, 'movies.dat'),
                                           sep='::', header=None, names=mnames, engine='python')
        self.ratings = pd.read_table(os.path.join(dataPath, 'ratings.dat'),
                                     sep='::', header=None, names=rnames, engine='python')

        # preprocess data
        self.preprocess()

        # make dataset
        self.rowData = None
        self.label = None
        self.makeDataset()

        # split dataset: train(70%) and test(30%)
        self.trainRowData = None
        self.trainLabel = None
        self.testRowData = None
        self.testLabel = None
        self.splitDataset()

        print("finish preparing data!")

    def preprocess(self):
        # ----------------------------------------- encode sparse feature ---------------------------------------------#
        # users: gender F/M -> 0/1, age 1/18/25/... -> 0/1/2/...
        lbe = LabelEncoder()
        self.userFeatures['gender'] = lbe.fit_transform(self.userFeatures['gender'])
        self.userFeatures['age'] = lbe.fit_transform(self.userFeatures['age'])

        # movies: genres genre1|genre2|... -> genre1, genre2,...,NonGenre -> 0/1/2/...
        maxLenGeners = ['genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6']
        self.movieFeatures[maxLenGeners] = 'NonGenre'
        # movies['len'] = 0
        for idx, genres in enumerate(self.movieFeatures['genres'].values):
            genreList = genres.split('|')
            for id, genre in enumerate(genreList):
                self.movieFeatures.loc[idx, 'genre{}'.format(id + 1)] = genre
            # self.movieFeatures['len'][idx] = len(genreList)

        encoderDict = {'NonGenre': 0, 'Action': 1, 'Adventure': 2, 'Animation': 3, "Children's": 4, 'Comedy': 5,
                       'Crime': 6,
                       'Documentary': 7, 'Drama': 8, 'Fantasy': 9, 'Film-Noir': 10, 'Horror': 11, 'Musical': 12,
                       'Mystery': 13, 'Romance': 14, 'Sci-Fi': 15, 'Thriller': 16, 'War': 17, 'Western': 18}
        for genre in maxLenGeners:
            for idx, genrex in enumerate(self.movieFeatures[genre].values):
                self.movieFeatures.loc[idx, genre] = encoderDict[genrex]

        # self.movieFeatures[maxLenGeners] = \
        #     lbe.fit_transform(self.movieFeatures[maxLenGeners].values.\
        #                       reshape(self.movieFeatures.shape[0]*len(maxLenGeners))).\
        #         reshape(self.movieFeatures.shape[0], -1)

        # ratings: rating 1/2/3/4/5 -> 0/0/0/1/1
        binE = Binarizer(threshold=self.ratingThreshold)
        self.ratings['rating'] = binE.fit_transform(self.ratings['rating'].values.reshape((-1, 1)))

        # -------------------------------------- drop features we don't use ------------------------------------------ #
        self.userFeatures = self.userFeatures.drop(columns='user_id', axis=1)
        self.userFeatures = self.userFeatures.drop(columns='zip', axis=1)

        self.movieFeatures = self.movieFeatures.drop(columns='title', axis=1)
        self.movieFeatures = self.movieFeatures.drop(columns='genres', axis=1)

        # ---------------------------------------- convert to numpy array -------------------------------------------- #
        self.userFeatures, self.movieFeatures, self.ratings = \
            map(lambda x: np.array(x, dtype=np.int64), [self.userFeatures, self.movieFeatures, self.ratings])

        # ----------------------------------------- fix users and movies --------------------------------------------- #
        self.userFeatures = np.insert(self.userFeatures, 0, values=0, axis=0)

        movies_ = np.zeros(shape=(self.movieFeatures[self.movieFeatures.shape[0] - 1, 0] + 1,
                                  self.movieFeatures.shape[1]), dtype=np.int64)
        movies_[self.movieFeatures[:, 0]] = self.movieFeatures
        self.movieFeatures = movies_

    def makeDataset(self):
        # group by users
        clusterByUser = []
        for userId in range(1, self.userFeatures.shape[0]):
            clusterByUser.append(self.ratings[np.where(self.ratings[:, 0] == userId)])

        # sort behaviors by timestamp
        for i in range(len(clusterByUser)):
            clusterByUser[i] = clusterByUser[i][np.argsort(clusterByUser[i][:, -1])]

        # generate dataset
        users = []
        behaviors = []
        ads = []
        labels = []
        for userRecoders in clusterByUser:
            hisBehaviors = []
            for i in range(userRecoders.shape[0]):
                users.append(userRecoders[i, 0])

                behaviorTmp = np.zeros(shape=(self.maxSequenceLen,), dtype=np.int64)
                if len(hisBehaviors) <= self.maxSequenceLen:
                    behaviorTmp[0: len(hisBehaviors)] = hisBehaviors
                else:
                    behaviorTmp[0: behaviorTmp.shape[0]] = \
                        hisBehaviors[len(hisBehaviors) - self.maxSequenceLen: len(hisBehaviors)]
                behaviors.append(behaviorTmp)

                ads.append(userRecoders[i, 1])
                labels.append(userRecoders[i, -2])

                if labels[-1] == 1:
                    hisBehaviors.append(userRecoders[i, 1])

        users, behaviors, ads, labels = map(lambda x: np.array(x, dtype=np.int64), [users, behaviors, ads, labels])
        users = np.expand_dims(users, 1)
        ads = np.expand_dims(ads, 1)

        self.rowData = np.hstack((users, behaviors, ads))
        self.label = labels

    def splitDataset(self):
        userRowData = []
        userLabel = []
        for userId in range(1, np.max(self.rowData[:, 0]) + 1):
            index = np.where(self.rowData[:, 0] == userId)

            if self.splitMethod == 'user' and \
                    (np.max(self.label[index]) != np.min(self.label[index])):  # drop users with whole 1/0 label
                userRowData.append(self.rowData[index])
                userLabel.append(self.label[index])

            userRowData.append(self.rowData[index])
            userLabel.append(self.label[index])

        if self.splitMethod == 'user':
            shuffleOrders = np.random.permutation(np.arange(len(userRowData)))
            splitPoint = np.int64(self.splitRation * len(userRowData))

            self.trainRowdata = [userRowData[shuffleOrders[k]] for k in range(splitPoint)]
            self.trainLabel = [userLabel[shuffleOrders[k]] for k in range(splitPoint)]
            self.trainRowData = np.vstack(self.trainRowdata)
            self.trainLabel = np.hstack(self.trainLabel)

            self.testRowData = [userRowData[shuffleOrders[k]] for k in range(splitPoint, len(shuffleOrders))]
            self.testLabel = [userLabel[shuffleOrders[k]] for k in range(splitPoint, len(shuffleOrders))]

        elif self.splitMethod == 'behavior':
            self.trainRowData = [userRowData[i][0: np.int64(userRowData[i].shape[0]*self.splitRation)]
                            for i in range(len(userRowData))]
            self.trainLabel = [userLabel[i][0: np.int64(userLabel[i].shape[0]*self.splitRation)]
                          for i in range(len(userLabel))]
            self.trainRowData = np.vstack(self.trainRowData)
            self.trainLabel = np.hstack(self.trainLabel)

            self.testRowData = [userRowData[i][np.int64(userRowData[i].shape[0]*self.splitRation):]
                                for i in range(len(userRowData))]
            self.testLabel = [userLabel[i][np.int64(userLabel[i].shape[0]*self.splitRation):]
                          for i in range(len(userLabel))]
        else:
            pass


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
    for fname in ratings:
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
    dataset = dataGenerator(dataPath=r"./dataset",
                            ratingBinThreshold=3, maxSequenceLen=10,
                            splitRatio=0.8,
                            splitMehtod='behavior')
    print(dataset.rowData)
    print(dataset.label)
    print(dataset.userFeatures)
    print(dataset.movieFeatures)

    print(dataset.trainRowData.shape)
    print(dataset.trainRowData)
    print(dataset.trainLabel.shape)
    print(dataset.trainLabel)

    print(len(dataset.testLabel))

    testRecode = 0
    for i in range(len(dataset.testRowData)):
        testRecode += dataset.testRowData[i].shape[0]
        if dataset.testRowData[i].shape[0] != dataset.testLabel[i].shape[0]:
            print("Error!")
    print(testRecode)
