import random

from DRIFT.DO import *
import pandas as pd

from utils.utils import train_part, GENRES, generate_symetric_key

DATASET_FOLDER = "datasets"

# TIME OK

class DatasetManager:
    """
        This class is the mother for the preprocessing, here we get how many users and items we want
    """

    def __init__(self, nb_user=None, nb_items=None):
        self.nb_user = nb_user
        self.nb_items = nb_items
        self.target = self.manage_dataset()
        self.dataset = "MovieLens"

    def preprocess(self):
        """
        :return: The datas with the users and items with the more interactions,
                         if this values is not given, we consider all the data
        """
        datas = pd.read_csv(self.target, sep=",", header=None)
        if self.nb_items is not None:
            items = datas[1].value_counts().index.tolist()[:self.nb_items]
            datas = datas.loc[datas[1].isin(items)]
        if self.nb_user is not None:
            user = datas[0].value_counts().index.tolist()[:self.nb_user]
            datas = datas.loc[datas[0].isin(user)]
        return datas

    def manage_dataset(self):
        """
        Here you need to manage your dataset such that :
            your new file is a tab separated list of user id | item id | rating | timestamp
            each one is separated by a column
        :return: the name of the file containing the datas
        """

        pass

    def preprocess_data_owner(self, key, nonce, nb_DO_max=None, data_train=None):
        print("entered")
        random.seed(123)
        data_genre = {}
        if nb_DO_max is None:
            all_genre_possible = GENRES
        else:
            all_genre_possible = list(range(nb_DO_max))

        for genre in all_genre_possible:
            data_genre[genre] = DataOwner(genre, key, nonce)
        items = list(set(data_train[1]))
        for id_item in items:
            df_item = data_train[data_train[1] == id_item]
            if len(df_item) == 0:
                continue
            if nb_DO_max is None:
                genres = self.get_genres(id_item)
                if genres is None:
                    continue
            else:
                genres = [random.randint(0, nb_DO_max - 1)]
            for genre in genres:
                data_genre[genre].add_item(id_item)

        return data_genre, max(items)

    def get_genres(self, id_item):
        pass


class ManageMovieLens(DatasetManager):
    def __init__(self):
        super().__init__()

    def manage_dataset(self, sep=None):
        file = 'ml-latest-small/ml_100_data'
        # file = 'ml-latest-small/ratings.csv'
        return f'{DATASET_FOLDER}/{file}'

    def preprocess(self):
        data = super().preprocess().iloc[1:].astype(float)
        genres = pd.read_csv(f"{DATASET_FOLDER}/ml-latest-small/movies.csv", sep=",", header=None).iloc[1:]
        genres = genres.astype({0: int})
        self.genres = genres
        data[2] = [int(float(value) >= 4) for value in data[2]]
        data = data.astype(int)
        data_test = pd.DataFrame()
        data_train = pd.DataFrame()
        for user in list(set(data[0])):
            d = data[data[0] == user].sort_values(by=[3])
            spliter = int(len(d) * train_part)
            data_train = data_train.append(d.iloc[:spliter])
            data_test = data_test.append(d.iloc[spliter:])
        return data_train, data_test

    def get_genres(self, id_item):
        df_item = self.genres[self.genres[0] == id_item]
        if len(df_item) == 0:
            return
        return list(df_item[2])[0].split("|")


class ManageMovieLensSmall(ManageMovieLens):
    def manage_dataset(self, sep=None):
        file = 'ml-latest-small/ratings.csv'
        return f'{DATASET_FOLDER}/{file}'


class ManageKASANDR(DatasetManager):
    def __init__(self):
        super().__init__()
        self.dataset = "KASANDR"

    def manage_dataset(self, file='KASANDR/train_KASANDR', sep=None):
        return f'{DATASET_FOLDER}/{file}'

    def preprocess(self):
        data_train = pd.read_csv(f"datasets/KASANDR/train_KASANDR.csv", sep=",", header=None).iloc[1:].astype(int)
        data_test = pd.read_csv(f"datasets/KASANDR/test_KASANDR.csv", sep=",", header=None).iloc[1:].astype(int)
        return data_train, data_test

    def get_genres(self, id_item):
        return random.sample(GENRES, random.randint(1, 5))

class ManagePANDOR(DatasetManager):
    def __init__(self):
        super().__init__()
        self.dataset = "PANDOR"

    def manage_dataset(self, file='PANDOR/items_filtered.csv', sep=None):
        return f'{DATASET_FOLDER}/{file}'

    def preprocess(self):
        data = super().preprocess().iloc[1:].astype(int)
        data_test = pd.DataFrame()
        data_train = pd.DataFrame()
        for user in list(set(data[0])):
            d = data[data[0] == user].sort_values(by=[3])
            spliter = int(len(d) * train_part)
            data_train = data_train.append(d.iloc[:spliter])
            data_test = data_test.append(d.iloc[spliter:])
        return data_train, data_test

    def get_genres(self, id_item):
        return random.sample(GENRES, random.randint(1, 5))

def preprocess(nonce, dataset, nb_DO_max=None):
    # preprocess
    MML = {
        "ML_small" : ManageMovieLensSmall,
        "ML": ManageMovieLens,
        "KASANDR": ManageKASANDR,
        "PANDOR": ManagePANDOR,

    }
    if dataset not in MML :
        raise NotImplementedError(f"\n{dataset} is not a valid dataset, \n "
                                  f"Datasets possible : \n"
                                  f"\t ML_small \n"
                                  f"\t ML \n"
                                  f"\t KASANDR \n"
                                  "\t PANDOR \n"
                                  )
    mml = MML[dataset]()
    print("Creating the testing and training data")
    data_train, data_test = mml.preprocess()
    print("Sending the data to the DO")
    key = generate_symetric_key()

    dos, nb_i = mml.preprocess_data_owner(key, nonce, nb_DO_max=nb_DO_max, data_train=data_train)
    print(data_train, data_test)
    print(dos)
    return (data_train, data_test, mml.dataset), dos, nb_i, key,
