import random


def to_id(data):
    n_u = []
    all_u = {}
    for u in data:
        if u not in all_u:
            all_u[u] = len(all_u) + 1
        n_u.append(all_u[u])
    return n_u


def create_new_dataset(PATH):
    df = pd.read_csv(PATH, sep=";", header=None).astype(str)
    new_df = pd.DataFrame({'userId': [], 'itemId': [], 'click': [], 'timestamp': []})
    new_df['userId'] = to_id(df[0])
    new_df['itemId'] = to_id(df[1])
    new_df['click'] = df[2]
    new_df['timestamp'] = df[3]
    new_df.to_csv("datasets/KASANDR/train.csv", index_label=False, index=False)
    print(df)


def filter_dataset(PATH):
    ratio = 2e6
    ttl = 0
    df = pd.read_csv(PATH + ".csv", sep=",", header=None).iloc[1:]
    users = list(set(df[0]))
    new_df_train = pd.DataFrame()
    for u in users:
        print(f"starting user {u} / 291485, ttl = {ttl} ")
        df_u = df[df[0] == u]
        if len(df_u) < 2 or len(df_u[df[2] == 0]) == 0 or len(df_u[df[2] == 1]) == 0:
            continue
        new_df_train = new_df_train.append(df_u)
        ttl += len(df_u[0])
        if ttl > ratio:
            break
    new_df_train[0] = to_id(new_df_train[0])
    new_df_train[1] = to_id(new_df_train[1])
    print(new_df_train, df)

def get_nb_user(nb, df):
    nb_u = list(map(lambda x : str(x), range(nb)))

    new_df = df[df["userId"].isin(nb_u)]

    print(new_df)
    new_df["itemId"] = to_id(new_df["itemId"])
    new_df.to_csv("datasets/PANDOR/items_filtered.csv", index_label=False, index=False)


if __name__ == "__main__":
    #df = pd.read_csv("datasets/PANDOR/items.csv", sep=",").astype(str)
    #filter_dataset("datasets/PANDOR/items")
    for i in range(0,20):
        print(chr(random.randint(50, 120)), end="")

