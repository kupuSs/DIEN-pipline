"""
@author kupuV
@time 20191003
"""
from tqdm import tqdm
import pandas as pd
import json
import datetime
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from joblib import Parallel, delayed
import multiprocessing
from data_process import config_amazon_books

random.seed(2019)

with open('../data_cache/itemID.json', 'r') as f:
    item2id = json.load(f)
item2id_reverse = dict(zip(item2id.values(), item2id.keys()))

with open('../data_cache/cateID.json', 'r') as f:
    cate2id = json.load(f)
cate2id_reverse = dict(zip(cate2id.values(), cate2id.keys()))


def get_input(df):
    return df


def run(row):
    neg_item_list = []
    for _ in range(len(row)):
        neg_item_index = random.randint(1, len(item2id_reverse) - 1)
        neg_item = item2id_reverse[neg_item_index]
        if neg_item not in row.split('|'):
            neg_item_list.append(neg_item)
    return neg_item_list


def tmp_func(dataframe):
    dataframe['neg_' + config_amazon_books.sequence_features[0]] = dataframe[config_amazon_books.sequence_features[0]].apply(run)
    return dataframe


def apply_parallel(dataframe_grouped, func):
    """利用 Parallel 和 delayed 函数实现并行运算"""
    results = Parallel(
        n_jobs=multiprocessing.cpu_count(),
        verbose=4,
        backend='multiprocessing'
    )(delayed(func)(group) for name, group in dataframe_grouped)
    return pd.concat(results)


if __name__ == '__main__':
    df = get_input()