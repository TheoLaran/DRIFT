import ast
import json
import os

import numpy as np


def res_to_JSON():
    target = "JSON_metrics"
    RES = "res"
    sep = ": "

    for datasets in os.listdir(RES):
        if datasets == target or os.path.isfile(f"{RES}/{datasets}"):
            continue
        print(datasets)
        all_results = {}
        for nb_run in os.listdir(f"{RES}/{datasets}"):
            if os.path.isfile(f"{RES}/{datasets}/{nb_run}") :
                continue
            for architecture in os.listdir(f"{RES}/{datasets}/{nb_run}"):
                if architecture == "DO" or not os.path.isfile(f"{RES}/{datasets}/{nb_run}/{architecture}"):
                    continue

                with open(f"{RES}/{datasets}/{nb_run}/{architecture}", "r") as result:
                    if os.path.isfile(f"{RES}/{target}/{datasets}_{nb_run}.json"):
                        with open(f"{RES}/{target}/{datasets}_{nb_run}.json", "r") as js_r:
                            text = js_r.read()
                            results = ast.literal_eval(text)[architecture]
                    else:
                        results = {}
                    temp = {}
                    while True:
                        l = result.readline()
                        if len(l) == 0:
                            break
                        line = l.split(sep)
                        categories = line[0].split('@')

                        if len(categories) != 2:
                            continue
                        categorie, number = categories
                        if categorie == "Mean Average Precision":
                            categorie = "MAP"
                        elif categorie == "precision":
                            categorie = "Precision"
                        if categorie not in temp:
                            temp[categorie] = []
                        value = float(line[1])
                        if value < 1:
                            value *= 100
                        temp[categorie].append(value)
                    for key in temp.keys() :
                        if key not in results :
                            results[key] = []
                        results[key].append(temp[key])
                all_results[architecture] = results
            with open(f"{RES}/{target}/{datasets}_{nb_run}.json", "w") as js:
                js.write(str(all_results).replace("'", '"'))






def JSON_to_latex(PATH):
    BACKSPACE = "\n" + r"\\ \hline" + "\n"
    with open(PATH, "r") as Json_read:
        txt = Json_read.read()
    data = json.loads(txt)
    methods = list(data.keys())
    compute = list(data[methods[0]].keys())

    res = r"\begin{tabular}"
    colmns = "|c||"
    for _ in range(len(methods)):
        colmns += "c|" * len(compute)
        colmns += "|"

    res += "{" + colmns[:-1] + "}"
    res += "\n" + r"\hline"

    #### ENTETE #####
    for i, method in enumerate(methods):
        if method == r"$\proto$" : method = "Main"
        colmns = "}{|c|}{"
        if i == 0:
            colmns = "}{|c||}{"

        res += "&\multicolumn{" + str(len(compute)) + colmns + method + "}"
    res += BACKSPACE

    res += "number of "
    cmpts = ""
    for c in compute:

        if c != "Precision":
            c = c.upper()
        cmpts += "&" + c

    res += cmpts * len(data)
    res += r"\\" + "\n"
    res += "items proposed" + ("& " * (len(compute) - 2) + r"& (in \%) " * 2) * len(methods)
    res += BACKSPACE

    temp = {}
    for m in methods:
        temp[m] = {}
        for c in compute :
            matrix = np.array(data[m][c])

            # 10 resuts for a metrics and a computation ( SAROS / MAIN / ... )
            method_result = np.mean(matrix, axis=0)
            temp[m][c] = method_result
    print(temp)

    for j in range(10):
        res += str(j + 1)
        # SAROS - Main

        # MAP / NDCG / ...
        for m in methods:
            for c in compute:
                res += str([f" & {round(temp[m][c][j], 2)}" ]).replace("[", "").replace("]", "").replace(",", " &").replace("'","")

        res += BACKSPACE

    res += r"\end{tabular}"

    print(res)


if __name__ == "__main__":
    res_to_JSON()
    #JSON_to_latex("res/JSON_metrics/MovieLens_0.json")
    JSON_to_latex("res/JSON_metrics/MovieLens_0.json")
