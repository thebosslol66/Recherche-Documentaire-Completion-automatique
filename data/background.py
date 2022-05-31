#!/usr/bin/env python3.10

"""
Regroup all functions for display and loading data easly.
It have functions to print levenstein table and jaro table.
It have too a function to compare all algorithms for distance editing
"""

import os, time, random, multiprocessing
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from IPython.display import display, HTML
from ipywidgets import widgets, interactive, fixed
import pandas as pd
import typing


DistanceFunctionVar = typing.TypeVar('DistanceFunctionVar', bound=typing.Callable[[str], typing.Callable[[str], float]])
PreTrainedPredictionVar =typing.TypeVar('PreTrainedPredictionVar', bound=typing.Callable[[typing.Callable[[str], typing.Callable[[str], float]], str, typing.Optional[str], typing.Optional[typing.Callable[[float], float]]], typing.Tuple[pd.DataFrame, float]])

custom_size = typing.TypeVar('custom_size', None, HTML)

azerty_around_letter = { "a":["z", "s", "q"],
                        "z": ["a", "e", "d", "s", "q"],
                        "e": ["z", "r", "f", "d", "s"],
                        "r": ["e", "t", "g", "f", "d"],
                        "t": ["r", "y", "h", "g", "f"],
                        "y": ["t", "u", "j", "h", "g"],
                        "u": ["y", "i", "k", "j", "h"],
                        "i": ["u", "o", "l", "k", "j"],
                        "o": ["i", "p", "m", "l", "k"],
                        "p": ["o", "m", "l"],
                        "q":["a", "z", "s", "x", "w"],
                        "s": ["z", "e", "d", "q", "x", "w"],
                        "d": ["e", "r", "f", "s", "x", "d"],
                        "f": ["e", "t", "g", "d", "c", "v"],
                        "g": ["r", "y", "h", "f", "v", "b"],
                        "h": ["t", "u", "j", "g", 'b', 'n'],
                        "j": ["y", "i", "k", "h", "j"],
                        "k": ["u", "o", "l", "j"],
                        "l": ["i", "p", "m", "k"],
                        "m": ["o", "p", "l"],
                        "w": ["q", "s", "x"],
                        "x": ["s", "d", "c", "w"],
                        "c": ["d", "f", "x", "v"],
                        "v": ["f", "g", "c", "b"],
                        "b": ["g", "h", "v", "n"],
                        "n": ["b", "h", "j"]
                       }
                              
def setWidgetTextSize(value :int) -> None:
    """
    Set the size of text widget to he a better render
    
    Parameters:
        value (int):
            0: Normal size
            1: Littel bigger for convinience
            2: The bigger for big letters
    """
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
        

def import_data(filename :str) -> pd.DataFrame:
    """
    Load sata from a csv file and get it as a pandas.DataFame.
    It shows an ipywidget.IntProgress during loading of data. 
    
    Parameters:
        filename (str):
            Retreive the data from the file

    Returns:
        (pd.DataFrame):
            The data from file name loaded in a pandas.DataFrame.
    """
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

def delete_useless_word_in_pd(data :pd.DataFrame) -> pd.DataFrame:
    """
    Delete lines wich contain multiple times the same word in the column 'ortho'
    and lines with ortho word length less than 3 chars
    
    Parameters:
        data (pandas.DataFrame):
            A dataframe containing a colulmn 'ortho'

    Returns:
        (pandas.DataFrame):
            The data without duplicate lines containing the same word in column 'ortho'
            and without word less than 3 Char
    """
    data = data[["ortho"]]
    data = data.drop_duplicates(["ortho"], keep="first").reset_index(drop=True)
    data = data[data["ortho"].str.len() > 2].reset_index(drop=True)
    return data




def chrono(fonction :typing.Callable[[list, dict], typing.Tuple[typing.Any, float]]):
    """
    Get time used for executing a function it returns it as a tuple of
    result of function and time elapsed
    """

    def wrapper(*args :list, **kwargs : dict) -> typing.Tuple[typing.Any, float]:
        temps_debut = time.time()
        y = fonction(*args, **kwargs)
        return y, time.time() - temps_debut

    return wrapper

def getPredictionsFromDict(word_dict :pd.DataFrame) -> PreTrainedPredictionVar:
    @chrono
    def getPredictions(funct :typing.Callable[[str], DistanceFunctionVar], word :str, start :str ="", mofier_funct : typing.Callable[[float], float] =lambda x: x) -> pd.DataFrame:
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


def getPredictionsFromDictForPreParamFunction(word_dict :pd.DataFrame) -> PreTrainedPredictionVar:
    @chrono
    def getPredictions(funct :DistanceFunctionVar, start :str="", mofier_funct : typing.Callable[[float], float] =lambda x: x) -> pd.DataFrame:
        start = start.lower()
        f = funct

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

def showLevensteinTable(title_text :str, c1 :str, c2 :str, data : typing.List[typing.List[int]]) -> None:
    """
    Show a levenstein table for word c1 and c2 with a table containing data
    
    Parameters:
        title_text (str):
            The title of the draw table 
        c1 (str):
            The word compared
        c2 (str):
            The word to compare
        data (list[list[int]]):
            The levenstein table between c1 and c2
    """

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
    ax.set_title(title_text)
    #plt.savefig(title_text.replace(" ", "_") + ".png", bbox_inches='tight')
    plt.show()


def showJaroTable(title_text :str, c1 :str, c2 :str, Winkler :bool =False, **kwargs :typing.Dict[str, typing.Any]) -> None:
    """
    Show a Jaro table for word c1 and c2
    It displays math associated
    
    Parameters:
        title_text (str):
            The title of the draw table 
        c1 (str):
            The word compared
        c2 (str):
            The word to compare
        Winkler (bool):
            If it needs to display calcs associated to the distance Jaro-Winkler
        kwargs (typing.Dict[str, AnyType]):
            Additional settings for the winkler calcs
            p (float):
                The number p used by buy the algorythm Jaro-Winkler
            threshold (float):
                The threshold used fo calculating the l when it's upper
            l (int):
                The length of common prefix
    """

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
    ax[0].set_title(title_text)
    #plt.savefig(title_text.replace(" ", "_") + ".png", bbox_inches='tight')
    plt.show()
    

def assign_label_to_func(getPredictions :PreTrainedPredictionVar, f :DistanceFunctionVar, rowNb :int) -> typing.Callable[[str, typing.Any], None]:
    """
    Assign a function to a group of label and apply a getPrediction function to search 3 best words
    
    Parameters:
        getPredictions (PreTrainedPredictionVar):
            The fonction getPrediction already configured with a dict
        f (DistanceFunctionVar):
            The function to calculate a distance between 2 word

    Returns:
        (typing.Callable[[str, typing.Any], None]):
            The fonction wich calculate the best 3 result for the algorithm and put it's result into the DictProxy.
    """
    modifier = lambda x: x
    if f.__name__ == "DistanceDeJaro" or f.__name__ == "DistanceDeJaroWinkler":
        modifier = lambda x: 1 - x

    def funct(word :str, return_dict :typing.Dict) -> None:
        """
        Calculate best 3 result for comparate word with all other word in preload function getPrediction and put result in the DictProxy
    
    Parameters:
        word (str):
            The word to seach the best same word in getPrediction
        return_dict (multiprocessing.managers.DictProxy):
            The dict to put results
    """
        if len(word) < 2:
            return

        res, time_elapsed = getPredictions(
            f, word=word, start=word[:2], mofier_funct=modifier
        )

        res = list(res["ortho"].values)
        res.append("{:.5f}".format(time_elapsed))
        return_dict[rowNb] = res

    return funct


def setup_autocomplete(getPredictions :PreTrainedPredictionVar, grid :widgets.GridspecLayout, algos :typing.Sequence[DistanceFunctionVar]) -> typing.Callable[[widgets.widget_string.Text], None]:
    funct_list = list()
    for index, algo in enumerate(algos):
        funct_list.append(assign_label_to_func(getPredictions, algo, index+1))

    shared_dict = multiprocessing.Manager().dict()

    last_send_value = widgets.widget_string.Text()

    def funct(sender :widgets.widget_string.Text) -> None:
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

def displayComparaisonInterface(getPredictions :PreTrainedPredictionVar, *args :typing.Sequence[DistanceFunctionVar]):
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
    text.on_submit(setup_autocomplete(getPredictions, grid, algos=args))
    text.add_class('custom-size')
    
    display(text)
    display(grid)

### Functions oganize text to data structure usable in python

def data_file_to_sentences(data :str) -> list:
    """
    Removes punctuation from a string and split into sentences
    
    Parameters:
        data (str):The string which is to be unpontuated.

    Returns:
        (list):The list of sentence terminate by a dot.
    """
    return (
        data.lower()
        .replace("...", ".")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("?", ".")
        .replace("!", ".")
        .replace('"', ".")
        .replace("‘", " ")
        .replace("-", " ")
        .replace("’", " ")
        .replace("'", " ")
        .replace(",", " ")
        .split(".")
    )

def remove_empty_words(l :list) -> list:
    """
    Removes empty strings in a list
    
    Parameters:
        l (list):The list with empty strings.

    Returns:
        (list):The list without empty strings.
    """
    return list(filter(lambda a: a != "", l))

def update_occ(d :dict, seq :str, w :str) -> None:
    """
    Append an occurence in the dict for the w after a seq.
    It increment it if the word has been already view after seq
    
    Parameters:
        d(dictr):
            The dict containing the data of all occurences.
        seq(str):
            The sequece for which a new word appears after
        w(str):
            The word which appear after the seq
    """
    if seq not in d:
        d[seq] = {}
    if w not in d[seq]:
        d[seq][w] = 0
    d[seq][w] = d[seq][w] + 1
    
def gen_random_from_tbl(t :dict) -> str:
    """
    Get a random word from a dict containing tuples with
    word and number of appearances.
    
    Parameters:
        t (dict):
            The the dict containing the words and their
            number of appearance in the training set.

    Returns:
        (str):The random word choosen.
    """
    return random.choices(list(t.keys()), weights=list(t.values()))[0]


def create_progress_bar_training_markov(sentences :list) -> widgets.IntProgress:
    """
    Create a progressbar for the data to train our markov chain model
    """
    nb_words = sum([len(remove_empty_words(i.split(" "))) for i in sentences])
    print(nb_words)
    pbar = widgets.IntProgress(
        value=0,
        min=0,
        max=nb_words,
        description="Create markov model:",
        bar_style="info",  # 'success', 'info', 'warning', 'danger' or ''
        style={"bar_color": "blue"},
        orientation="horizontal",
    )
    display(pbar)
    return pbar