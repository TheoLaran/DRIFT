import json
import ast
import os


GENRES = ['Main', 'Animation', 'Sci-Fi', 'Thriller', 'Comedy', 'Crime', 'Drama', 'Adventure', 'Children', 'Action', 'Romance',
          'War', 'Western', 'Horror', 'Musical', 'IMAX', 'Mystery', 'Fantasy', 'Documentary',
          '(no genres listed)']

figs = []

def write_prediction(content, genre=None, type=None):
    if genre is None or type is None :
        print("Nothing to write")
        return
    with open(f"res/{type}_{genre}", "w") as file:
        file.write(content.replace('[', '').replace(']', ''))


def read(JSON_PATH, init):
    with open(JSON_PATH, "r") as Json_read:
        txt = Json_read.read()
        if init or len(txt) == 0:
            return {
                "train": '[]',
                "test": '[]',
                "time": '[]',
                "seconds": '[]'
            }
    return json.loads(txt)


def manage(s):
    st = s.replace('\n', '')
    return f"[{st}]"


def add_data(data, cat, st):
    t = ast.literal_eval(data[cat])
    t.append(st)
    return t


def write(genre, *args, init=False):
    JSON_PATH = f"JSON/{genre}.json"
    data = read(JSON_PATH, init)
    if init :
        with open(JSON_PATH, "w") as Json_write:
            Json_write.write(str(data).replace("'", '"'))
        return
    if len(args) < 4 :
        raise Exception(f"Should have at least 4 arguments : train, time, test, nb_blocks \n Got only {len(args)} elements")

    train, test, blocks, time = args

    data["train"] = add_data(data, "train", train)
    data["test"] = add_data(data, "test", test)
    data["time"] = add_data(data, "time", blocks)
    data["seconds"] = add_data(data, "seconds", time)

    with open(JSON_PATH, "w") as Json_write:
        res = ""
        for key in data.keys():
            r = f'"{key}" : {data[key]}, '
            print(r)
            res += r.replace('"','')
        Json_write.write(res.replace("'", '"'))




def reset():
    for files in os.listdir("JSON"):
        genre = files.replace(".json", "")
        write(genre, init=True)


if __name__ == "__main__":
    reset()






