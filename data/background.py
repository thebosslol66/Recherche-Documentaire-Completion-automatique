import time
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML
from ipywidgets import widgets, interact, interactive, fixed, interact_manual
import pandas as pd
import os
import multiprocessing

custom_size = None

def setWidgetTextSize(value):
    global custom_size
    if value == 0:
        font_size = "15px"
    elif value == 1:
        font_size = "20px"
    else:
        font_size = "25px"
    html_value = """<style> .custom-size, .custom-size input[type="text"] 
            {
                text-align: center;
                width: auto;
                height:auto;
                font-size: """+ font_size +""";
                white-space: break-spaces;
            } 
            </style>"""
    if custom_size is None:
        custom_size = HTML(html_value)
        display(custom_size)
    else:
        custom_size.value = html_value
        

def import_data(filename):
    temp = pd.read_csv(filename, nrows=20)
    N = len(temp.to_csv(index=False))
    df = [temp[:0]]
    t = int(os.path.getsize(filename) / N * 20 / 10**5) + 1

    pbar = widgets.IntProgress(
        value=0,
        min=0,
        max=t,
        description="Loading:",
        bar_style="",  # 'success', 'info', 'warning', 'danger' or ''
        style={"bar_color": "blue"},
        orientation="horizontal",
    )

    display(pbar)

    for i, chunk in enumerate(pd.read_csv(filename, chunksize=10**5, low_memory=False)):
        df.append(chunk)
        pbar.value += 1

    res = pd.concat(df)
    del df, temp
    return res

def delete_useless_word_in_pd(data):
    data = data[["ortho"]]
    data = data.drop_duplicates(["ortho"], keep="first").reset_index(drop=True)
    data = data[data["ortho"].str.len() > 2].reset_index(drop=True)
    return data




def chrono(fonction):
    """Chronométre le temps mis par la fonction"""

    def wrapper(*args, **kwargs):
        temps_debut = time.time()
        y = fonction(*args, **kwargs)
        return y, time.time() - temps_debut

    return wrapper

def getPredictionsFromDict(word_dict):
    @chrono
    def getPredictions(funct, word, start="", mofier_funct=lambda x: x):
        word = word.lower()
        start = start.lower()
        f = funct(word)

        mask = word_dict["ortho"].str.startswith(start) == True
        word_dict_affinity = word_dict[
            mask
        ].apply(
            lambda x: pd.Series(
                [x["ortho"], mofier_funct(f(x["ortho"]))], index=["ortho", "affinity"]
            ),
            axis=1,
        )

        results = word_dict_affinity.nsmallest(3, "affinity", keep="first")
        return results
    return getPredictions

def showLevensteinTable(title_text, c1, c2, data):

    column_headers = list(" " + c1)
    row_headers = list(" " + c2)

    ncol = len(column_headers)
    nrow = len(row_headers)

    fig, ax = plt.subplots(figsize=((ncol + 1) / 2, (nrow + 1) / 2), dpi=100)

    # draw grid lines
    ax.plot(
        np.tile([0, ncol + 1], (nrow + 2, 1)).T,
        np.tile(np.arange(nrow + 2), (2, 1)),
        "k",
        linewidth=1,
    )
    ax.plot(
        np.tile(np.arange(ncol + 2), (2, 1)),
        np.tile([0, nrow + 1], (ncol + 2, 1)).T,
        "k",
        linewidth=1,
    )

    # plot labels
    for icol, col in enumerate(column_headers):
        ax.text(icol + 1.5, nrow + 0.5, col, ha="center", va="center")
    for irow, row in enumerate(row_headers):
        ax.text(0.5, nrow - irow - 0.5, row, ha="center", va="center")

    ax.add_patch(mpatches.Rectangle((1, nrow), ncol, 1, alpha=1, facecolor="cyan"))
    ax.add_patch(mpatches.Rectangle((0, 0), 1, nrow, alpha=1, facecolor="cyan"))

    # plot table content
    for irow, row in enumerate(data):
        for icol, cell in enumerate(row):
            ax.text(icol + 1.5, nrow - irow - 0.5, cell, ha="center", va="center")

    ax.axis([-0.5, ncol + 1.5, -0.5, nrow + 1.5])

    x = len(data[0]) - 1
    y = len(data) - 1

    while x > 0 or y > 0:
        x_end_arrow = x
        y_end_arrow = y

        c = "red"

        if x == 0:
            y -= 1
        elif y == 0:
            x -= 1
        else:
            m = min(data[y - 1][x], data[y][x - 1], data[y - 1][x - 1])
            if (
                x > 0
                and y > 0
                and c1[x - 1] == c2[y - 2]
                and c1[x - 2] == c2[y - 1]
                and data[y - 1][x - 1] == data[y][x]
                and c1[x - 2] != c1[x - 1]
            ):
                c = "green"
                x -= 2
                y -= 2
            elif m == data[y - 1][x - 1]:
                x -= 1
                y -= 1
            elif m == data[y - 1][x]:
                y -= 1
            else:
                x -= 1

        x_begin_pos = x if x == x_end_arrow else x + 0.25
        y_begin_posy = y if y == y_end_arrow else y + 0.25
        x_len = x_end_arrow - x if x == x_end_arrow else x_end_arrow - x - 0.25*2
        y_len = - (y_end_arrow - y) if y == y_end_arrow else -(y_end_arrow - y - 0.25*2)
        
        ax.arrow(x_begin_pos + 1.5, nrow - y_begin_posy - 0.5, x_len, y_len,
            color=c,
            linewidth=3,
            alpha=0.5,
        )

    plt.axis("off")
    #ax.set_title(title_text)
    plt.savefig(title_text.replace(" ", "_") + ".png")
    plt.show()


def showJaroTable(title_text, c1, c2, Winkler=False, **kwargs):

    column_headers = list(c1)
    row_headers = list(c2)

    ncol = len(column_headers)
    nrow = len(row_headers)

    fig, ax = plt.subplots(
        1,
        2,
        figsize=((ncol + 1) / 2 + 3, (nrow + 1) / 2),
        dpi=100,
        gridspec_kw={"width_ratios": [(ncol + 1) / 2, 3]},
    )

    # draw grid lines
    ax[0].plot(
        np.tile([0, ncol + 1], (nrow + 2, 1)).T,
        np.tile(np.arange(nrow + 2), (2, 1)),
        "k",
        linewidth=1,
    )
    ax[0].plot(
        np.tile(np.arange(ncol + 2), (2, 1)),
        np.tile([0, nrow + 1], (ncol + 2, 1)).T,
        "k",
        linewidth=1,
    )

    # plot labels
    for icol, col in enumerate(column_headers):
        ax[0].text(icol + 1.5, nrow + 0.5, col, ha="center", va="center")
    for irow, row in enumerate(row_headers):
        ax[0].text(0.5, nrow - irow - 0.5, row, ha="center", va="center")

    ax[0].add_patch(mpatches.Rectangle((1, nrow), ncol, 1, alpha=1, facecolor="cyan"))
    ax[0].add_patch(mpatches.Rectangle((0, 0), 1, nrow, alpha=1, facecolor="cyan"))

    len_c1 = len(c1)
    len_c2 = len(c2)
    match = 0

    limit = int(max(len_c1, len_c2) / 2) - 1

    table_match = []
    for i in range(len_c2):
        table_match.append([0] * len_c1)

    for i in range(len_c1):
        for j in range(max(0, i - limit), min(len_c2, i + limit + 1)):

            ax[0].add_patch(
                mpatches.Rectangle(
                    (i + 1, nrow - (j + 1)), 1, 1, alpha=1, facecolor="yellow"
                )
            )

            if c1[i] == c2[j] and not (1 in table_match[j]) and (not 1 in [table_match[k][i] for k in range(len_c2)]):
                table_match[j][i] = 1

    # plot table content
    for irow, row in enumerate(table_match):
        for icol, cell in enumerate(row):
            ax[0].text(icol + 1.5, nrow - irow - 0.5, cell, ha="center", va="center")

    ax[0].axis([-0.5, ncol + 1.5, -0.5, nrow + 1.5])

    c1_match = [0] * len_c1
    c2_match = [0] * len_c2

    for i in range(len_c1):
        for j in range(max(0, i - limit), min(len_c2, i + limit + 1)):

            if c1[i] == c2[j] and not c2_match[j]:
                c1_match[i] = 1
                c2_match[j] = 1
                match += 1
                break

    t = 0
    point = 0

    for i in range(len_c1):
        if c1_match[i]:
            while not c2_match[point]:
                point += 1
            if c1[i] != c2[point]:
                t += 1
            point += 1
    t /= 2

    result = 0
    if match != 0:
        result = (
            (match / float(len_c1))
            + (match / float(len_c2))
            + ((match - t) / float(match))
        ) / 3.0

    formules = [
        r"$d_j = \frac{1}{3}\left(\frac{m}{len(c1)}+\frac{m}{len(c2)}+\frac{m-t}{m}\right)$",
        r"$len(c1) = {}$, $len(c2) = {}$".format(len_c1, len_c2),
        r"$m = {}$, $t = {}$".format(match, int(t)),
        r"$d_j = \frac{{1}}{{3}}\left(\frac{{{m}}}{{{c1}}}+\frac{{{m}}}{{{c2}}}+\frac{{{m}-{t}}}{{{m}}}\right) = {r:.4f}$".format(
            m=match, c1=len_c1, c2=len_c2, t=int(t), r=result
        ),
    ]

    if Winkler:
        p = kwargs.get("p", 0.1)
        threshold = kwargs.get("threshold", 0.7)
        l = kwargs.get("l", 1)
        r2 = result + (l * p * (1 - result))
        formules.append("Winkler:")
        formules.append(r"p={}, l=${}$, threshold=${}$".format(p, l, threshold))
        formules.append(r"$d_w = d_j +(lp(1-d_j))$")
        formules.append(
            "$d_j={0:.4f} + ({1} * {2} * (1-{0:.4f})) = {3:.4f}$".format(
                result, l, p, r2
            )
        )

    for i, formule in enumerate(formules):
        ax[1].text(0, 0.9 - i * 0.1, formule)

    ax[0].axis("off")
    ax[1].axis("off")
    #ax[0].set_title(title_text)
    plt.savefig(title_text.replace(" ", "_") + ".png")
    plt.show()
    

### Fonctions to display and show comparaison between results of different algorithms
def assign_label_to_func(f, rowNb):
    modifier = lambda x: x
    if f.__name__ == "DistanceDeJaro" or f.__name__ == "DistanceDeJaroWinkler":
        modifier = lambda x: 1 - x

    def funct(word, return_dict):

        if len(word) < 2:
            return

        res, time_elapsed = getPredictions(
            f, word=word, start=word[:2], mofier_funct=modifier
        )

        res = list(res["ortho"].values)
        res.append("{:.5f}".format(time_elapsed))
        return_dict[rowNb] = res

    return funct


def setup_autocomplete(grid, algos):
    funct_list = list()
    for index, algo in enumerate(algos):
        funct_list.append(assign_label_to_func(algo, index+1))

    shared_dict = multiprocessing.Manager().dict()

    last_send_value = widgets.widget_string.Text()

    def funct(sender):
        v = sender.value.split(' ')[-1]
        if last_send_value.value == v:
            return

        last_send_value.value = v

        process_list = list()
        for f in funct_list:
            process_list.append(
                multiprocessing.Process(target=f, args=(v, shared_dict))
            )

        for p in process_list:
            p.start()

        for p in process_list:
            p.join()

        for key, values in shared_dict.items():
            for i in range(len(values)):
                grid[i + 1, key].value = values[i]

    return funct

def displayComparaisonInterface(*args):
    grid = widgets.GridspecLayout(len(args)+1, 5)
    
    def formatAlgoName(funct):
        import regex as re
        name = funct.__name__
        name = name.replace("DistanceDe", '')
        name = re.sub(r"([^&])([A-Z])", r"\1-\2", name)
        return name

    algo_name = [formatAlgoName(funct) for funct in args]
    
    for i in range(5):
        for j in range(5):
            text = ""
            if j == 0:
                text = str(i)
                if i == 0:
                    continue
                if i == 4:
                    text = "Time elapsed"
            else:
                if i == 0:
                    text = algo_name[j - 1]
            grid[i, j] = widgets.Label(text, height="auto", width="auto")
            grid[i, j].add_class('custom-size')
            
    text = widgets.Text("")
    text.on_submit(setup_autocomplete(grid, algos=args))
    text.add_class('custom-size')
    
    display(text)
    display(grid)
