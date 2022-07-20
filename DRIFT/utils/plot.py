import ast
import json
import os

import numpy as np
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as pl

colors = ["blue", "orange", "green", "red", "purple", "yellow"]
COLORS = iter(colors)


def aggreg(l, f):
    s = min([len(ds) for ds in l])

    return np.array([f([m[i] for m in l]) for i in range(s)])


def subplot(v, plt, x, label, color, test=False):
    sd = aggreg(v, np.std)
    v = aggreg(v, np.mean)
    m = min(len(v), len(x))
    x = x[:m]
    v = v[:m]
    sd = sd[:m]
    mn = v - sd
    mx = v + sd
    if not test :
        plt.plot(x, v, "--", label=label, color=color, markevery=v)
    else :
        plt.plot(x, v, label=label, color=color, markevery=v)
    plt.fill_between(x, mn, mx, color=color, alpha=0.2)


def _plot_time(losses, x, test_losses):
    pl.grid(True)
    # x = aggreg(x, np.mean)
    x = x[-1]
    subplot(losses, pl, x, 'Train', "blue")
    subplot(test_losses, pl, x, 'Test', "Orange")
    pl.xlabel('Time')
    pl.ylabel('Loss')
    pl.legend()


def get_elems(file):
    with open(file, "r") as Json_read:
        txt = Json_read.read()
    data = json.loads(txt)
    return ast.literal_eval(data["train"]), ast.literal_eval(data["time"]), data["seconds"], ast.literal_eval(data["test"])


def axis(a_x, mn_time, mx_time, mn_y, mx_y, ttl):
    a_x.set_xlim([mn_time, mx_time])
    a_x.set_ylim([mn_y, mx_y])
    a_x.set_title(ttl)


def main(args, scale_epoch, out=None):
    target = [st for st in args]
    mn_time, mn_y, mx_y = 0, 0.5, 0.7
    mx_time = 50
    target_loss = []
    target_times = []
    target_train = []
    CLR = []
    names = []
    for filename in os.listdir("JSON"):
        PATH = f"JSON/{filename}"
        if not os.path.isfile(PATH) : continue
        end = filename.replace(".json", "")
        if "_secure" not in end:
            continue
        is_target = list(filter(lambda x: x in end, target))
        losses, blocks, times, test_losses = get_elems(PATH)
        if len(times) == 0:
            print(end)
            continue
        if "Main" in end :
            times = [res["Update"] for res in times]

        #times = [list(map(lambda x : x / 1000, times[-1]))]


        if scale_epoch:
            #mx_time = min([len(time) for time in times])
            title = "EPOCH"
            blocks = [list(range(1, mx_time + 1))] * len(blocks)
        else :
            title = 'Blocks completed (x 10^3) '
            blocks = [list(map(lambda x : x * 1e-3, blocks[0]))]
        if is_target:
            target_train.append(losses)
            target_times.append(blocks)
            target_loss.append(test_losses)
            CLR.append(next(COLORS))
            if end == "Animation" :
                names.append("One DO")
            else :
                end = end.split("_")[0]
                if end == "Main" : end = "DRIFT"
                names.append(end)

        fg = pl.figure()
        ax = fg.add_axes([0.1, 0.1, 0.8, 0.8])
        axis(ax, mn_time, mx_time, mn_y, mx_y, end)
        _plot_time([losses[-1]], [times[-1]], [test_losses[-1]])
        fg.savefig(f"plot/{end}.pdf")
        pl.close(fg)



    fg = pl.figure()
    ax = fg.add_axes([0.1, 0.1, 0.8, 0.8])
    mn_time = 1
    mx_time = [5]

    axis(ax, mn_time, max(mx_time), 0.5, 0.7, "ML 1M")
    pl.grid(True)
    pl.xlabel(title)
    pl.ylabel('Loss')
    for color, (name, (loss, (train,times))) in zip(CLR, zip(names, zip(target_loss, zip(target_train, target_times)))):
        x = times[-1]
        subplot(train, pl, x, "Train " + name, color)
        subplot(loss, pl, x, "Test " + name, color, test=True)
        pl.legend(loc="lower left")
    if out is None :
        out = f"plot/{target}_train.pdf"
    fg.savefig(out)

def create_leg(cat, name):
    return f'{name} {round(cat * 100, 2)} %'
def time(out=None):

    if out is None:
        out = "plot/time.pdf"
    pl.figure(figsize=(8, 8))
    _, _, times, _ = get_elems( f"JSON/Main_secure.json")
    legend = []
    times = times[-1]
    a = [times[key] if key != "Update" else times["Update"][1] for key in times.keys()]
    x = []
    all_dos = [times[key] if "DO" in key else 0 for key in times.keys()]
    ttl = sum(a)
    theme = pl.get_cmap('copper')
    theme_not_DO = pl.get_cmap('hsv')
    colors = []
    i_do = 0
    i_other = 0
    for i, key in enumerate(times.keys()):
        if key == "Update" :
            val = times["Update"][1]
        else:
            val = times[key]
        x.append(val / ttl)
        if "DO" in key :
            colors.append(theme((21 - i_do) / 42 + 0.5))
            i_do += 1
            continue
        colors.append(theme_not_DO(i_other / (len(times.keys()) - 21)))
        i_other += 1
        legend.append(create_leg(x[-1], key))
    legend.append(create_leg(sum(all_dos) / ttl, "DO"))
    x.append(0)
    colors.append(theme(0.5))

    pl.pie(x,
           pctdistance = 0.7, labeldistance = 1.4, colors=colors)
    pl.legend(legend)
    pl.savefig(out)

def test(PATH, compute, out=None, mx=5000, label=""):
    if out is None :
        out = "plot/train.pdf"
    pas = 1
    to_take = list(range(1, 11, pas))
    to_take = list(map(lambda x: x + 0.025, to_take))
    with open(PATH, "r") as Json_read:
        data = Json_read.read()
    data = ast.literal_eval(data)
    fg = pl.figure()
    ax = fg.add_axes([0.1, 0.1, 0.8, 0.8])



    pl.yticks(range(11))
    axis(ax, 68, mx, 0, 11, label)
    ax.set_ylabel("Nb of item proposed")

    ax.set_xlabel("Score" + " (in %)" if mx <= 100 else "Precision")
    for method in data:
        matrix = data[method][compute]
        value = np.mean(matrix, axis=0)
        v = [value[i] for i in range(len(to_take))]

        pl.barh(to_take, v, height=0.3 * (-1 if method == "Main" else 1), align='edge', label=(method if method == "SAROS" else "DRIFT"))
        to_take = list(map(lambda x : x - 0.05, to_take))

    pl.legend(loc="upper right")
    fg.savefig(out)




if __name__ == "__main__":

    PATH_OUT = "plot"
    main(sys.argv[1:-1], int(sys.argv[-1]), out=f'{PATH_OUT}/res_train.pdf')
    for i in [1, 2, 3] :
        for j in ["MAP", "ndcg", "Precision"] :
            is_pre = j == "Precision"
            out = f'{PATH_OUT}/res_{("pcs" if is_pre else j).lower()}_{i}.pdf'
            test(f"res/JSON_metrics/MovieLens_{(i - 1) * 2}.json", j, out=out, mx=5000 if is_pre else 90, label=f"{j.upper()} score after {i * 4 - 3} epoch ")
    time(f'{PATH_OUT}/time.pdf')

"""

Res ML1M main {"train": "[[0.690439, 0.65236765, 0.6378051]]", "test": "[[0.6656974, 0.6378415, 0.6225581]]", "time": "[[0, 446562, 893124]]", "seconds": "[[0, 7888.076644659042, 17283.373960018158]]"}
Res ML1M SAROS {"train": "[[0.6846176846796135, 0.6447090946700281, 0.6290737585556787]]", "test": "[[0.66390055, 0.6371347, 0.6230434]]", "time": "[[0, 468813, 937626]]", "seconds": "[[11828.768617868423, 23266.770466804504, 33735.815615177155]]"}

"""

