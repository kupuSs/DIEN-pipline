"""
@author kupuV
@time 20191003
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import datetime
import random
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

random.seed(2019)


def truncated_sequence(df, sequence_features, limit=20):
    for col in tqdm(sequence_features, desc='sequence_truncated'):
        df[col] = df[col].apply(lambda x: '|'.join(x.split('|')[:limit]))
    return df


def encoder_sparse_feature_train(df, sparse_features):
    for col in tqdm(sparse_features, desc='encoder for train sparse_features'):
        cache_sf = {'UNK': 0}
        idx = 1
        for value in df[col]:
            if value not in cache_sf:
                cache_sf[value] = idx
                idx += 1
        with open('../data_cache/' + col + '.json', 'w') as f:
            json.dump(cache_sf, f)
        df[col] = df[col].map(cache_sf)
    with open('../data_cache_date.txt', 'w') as f:
        f.write('data_cache for ' + str(datetime.date.today()) + '\n')
    return df


def encoder_sparse_feature_test(df, sparse_features):
    for col in tqdm(sparse_features, desc='encoder for test sparse_features'):
        with open('../data_cache/' + col + '.json', 'r') as f:
            cache_sf = json.load(f)
        df[col] = df[col].map(cache_sf)
        df[col].fillna(0, inplace=True)

    return df


def negative_sampling(df, sequence_features, features_type='amazon_books', path='../data_cache/',
                      behavior_features1='itemID', behavior_features2='cateID',
                      is_training=True):
    """
    :param is_training:
    :param behavior_features1: special features(指定的特征，即需要negative sampling的特征)
    :param behavior_features2: special features(指定的特征，即需要negative sampling的特征)
    :param df:
    :param sequence_features:
    :param features_type: 不同的特征体系要用不同的方式
    :param path:
    :return:
    """
    if features_type is 'amazon_books':
        with open(path + behavior_features1 + '.json', 'r') as f:
            item2id = json.load(f)
        item2id_reverse = dict(zip(item2id.values(), item2id.keys()))

        with open(path + behavior_features2 + '.json', 'r') as f:
            cate2id = json.load(f)
        cate2id_reverse = dict(zip(cate2id.values(), cate2id.keys()))

        if is_training:
            mid_cat = {}
            for i in tqdm(range(len(df)), desc='extract ' + behavior_features1 + '2' + behavior_features2 + '(amazon_books)'):
                if str(df[behavior_features1][i]) not in mid_cat:
                    mid_cat[str(item2id_reverse[df[behavior_features1][i]])] = str(cate2id_reverse[df[behavior_features2][i]])
            with open('../data_cache/mid2cat.json', 'w') as f:
                json.dump(mid_cat, f)
        else:
            with open('../data_cache/mid2cat.json', 'r') as f:
                mid_cat = json.load(f)

        def run(row):
            neg_item_list = []
            for _ in range(len(row)):
                neg_item_index = random.randint(1, len(item2id_reverse) - 1)
                neg_item = item2id_reverse[neg_item_index]
                if neg_item not in row.split('|'):
                    neg_item_list.append(neg_item)
            return neg_item_list

        df['neg_' + sequence_features[0]] = df[sequence_features[0]].apply(run)
        df['neg_' + sequence_features[1]] = df['neg_' + sequence_features[0]].apply(
            lambda x: list(map(lambda y: mid_cat.get(y, 'UNK'), x)))

        def list2str(row):
            res = ''
            for i in row:
                res = res + str(i) + '|'
            return res[:-1]

        df['neg_' + sequence_features[0]] = df['neg_' + sequence_features[0]].apply(list2str)
        df['neg_' + sequence_features[1]] = df['neg_' + sequence_features[1]].apply(list2str)

        return df, sequence_features + ['neg_' + sequence_features[0], 'neg_' + sequence_features[1]]
    elif features_type is 'taobao':
        items_info = pd.read_csv(
            '../data/taobao_items',
            sep='\t',
            names=["itemID", "class1ID", "classID", "brandID", "price"])

        if is_training:
            vv = 'train'
        else:
            vv = 'test'

        for col in tqdm(['itemID', 'classID'], desc='encoder for items_side(neg_sample_taobao)' + '_' + vv):
            cache_sf = {'UNK': 0}
            idx = 1
            for value in items_info[col]:
                if value not in cache_sf:
                    cache_sf[value] = idx
                    idx += 1
            items_info[col] = items_info[col].map(cache_sf)
            if col == 'itemID':
                item2id_reverse = dict(zip(cache_sf.values(), cache_sf.keys()))

        if is_training:
            mid_cat = {}
            for i in tqdm(range(len(items_info)), desc='extract ' + behavior_features1 + '2' + behavior_features2):
                if str(items_info['itemID'][i]) not in mid_cat:
                    mid_cat[str(items_info['itemID'][i])] = str(items_info['classID'][i])
            with open('../data_cache/mid2cat.json', 'w') as f:
                json.dump(mid_cat, f)
        else:
            with open('../data_cache/mid2cat.json', 'r') as f:
                mid_cat = json.load(f)

        def list2str(row):
            res = ''
            for i in row:
                res = res + str(i) + '|'
            return res[:-1]

        # df[sequence_features[0]] = df[sequence_features[0]].apply(list2str)
        # df[sequence_features[1]] = df[sequence_features[1]].apply(list2str)

        def run(row):
            neg_item_list = []
            for _ in range(len(row)):
                neg_item_index = random.randint(1, len(item2id_reverse) - 1)
                neg_item = item2id_reverse[neg_item_index]
                if neg_item not in row.split('|'):
                    neg_item_list.append(neg_item)
            return neg_item_list

        df['neg_' + sequence_features[0]] = df[sequence_features[0]].apply(run)
        df['neg_' + sequence_features[1]] = df['neg_' + sequence_features[0]].apply(
            lambda x: list(map(lambda y: mid_cat.get(y, 'UNK'), x)))

        df['neg_' + sequence_features[0]] = df['neg_' + sequence_features[0]].apply(list2str)
        df['neg_' + sequence_features[1]] = df['neg_' + sequence_features[1]].apply(list2str)

        return df, sequence_features + ['neg_' + sequence_features[0], 'neg_' + sequence_features[1]]
    else:
        pass


def encoder_sequence(df, sequence_features, features_type='amazon_books', path='../data_cache/',
                     behavior_features1='itemID', behavior_features2='cateID', sequence_length=20):
    """
    :param df:
    :param sequence_features:
    :param features_type:
    :param path:
    :param behavior_features1:
    :param behavior_features2:
    :return:
    """
    if features_type is 'amazon_books':
        with open(path + behavior_features1 + '.json', 'r') as f:
            item2id = json.load(f)

        with open(path + behavior_features2 + '.json', 'r') as f:
            cate2id = json.load(f)

        def match_item(x):
            key_ans = x.split('|')
            return list(map(lambda x: item2id.get(x, 0), key_ans))  # 填0问题不大但不是最佳

        hist_item_list = list(map(match_item, df[sequence_features[0]].values))
        hist_item_len = list(map(len, hist_item_list))
        hist_item_list = pad_sequences(hist_item_list, maxlen=sequence_length, padding='post')

        neg_hist_item_list = list(map(match_item, df[sequence_features[2]].values))
        neg_hist_item_list = pad_sequences(neg_hist_item_list, maxlen=sequence_length, padding='post')

        def match_cate(x):
            key_ans = x.split('|')
            return list(map(lambda x: cate2id.get(x, 0), key_ans))  # 填0问题不大但不是最佳

        hist_cate_list = list(map(match_cate, df[sequence_features[1]].values))
        hist_cate_len = list(map(len, hist_cate_list))
        hist_cate_list = pad_sequences(hist_cate_list, maxlen=sequence_length, padding='post', )

        neg_hist_cate_list = list(map(match_cate, df[sequence_features[3]].values))
        neg_hist_cate_list = pad_sequences(neg_hist_cate_list, maxlen=sequence_length, padding='post', )

        return hist_item_list, neg_hist_item_list, hist_item_len, hist_cate_list, neg_hist_cate_list, hist_cate_len
    elif features_type is 'taobao':
        items_info = pd.read_csv(
            '../data/taobao_items',
            sep='\t',
            names=["itemID", "class1ID", "classID", "brandID", "price"])

        for col in tqdm(['itemID', 'classID'], desc='encoder for items_side(encoder_sequence_taobao)'):
            cache_sf = {'UNK': 0}
            idx = 1
            for value in items_info[col]:
                if value not in cache_sf:
                    cache_sf[value] = idx
                    idx += 1
            items_info[col] = items_info[col].map(cache_sf)
            if col == 'itemID':
                item2id = cache_sf
            elif col == 'classID':
                class2id = cache_sf

        def match_item(x):
            key_ans = x.split('|')
            return list(map(lambda x: item2id.get(x, 0), key_ans))  # 填0问题不大但不是最佳

        hist_item_list = list(map(match_item, df[sequence_features[0]].values))
        hist_item_len = list(map(len, hist_item_list))
        hist_item_list = pad_sequences(hist_item_list, maxlen=sequence_length, padding='post')

        neg_hist_item_list = list(map(match_item, df[sequence_features[2]].values))
        neg_hist_item_list = pad_sequences(neg_hist_item_list, maxlen=sequence_length, padding='post')

        def match_cate(x):
            key_ans = x.split('|')
            return list(map(lambda x: class2id.get(x, 0), key_ans))  # 填0问题不大但不是最佳

        hist_cate_list = list(map(match_cate, df[sequence_features[1]].values))
        hist_cate_len = list(map(len, hist_cate_list))
        hist_cate_list = pad_sequences(hist_cate_list, maxlen=sequence_length, padding='post', )

        neg_hist_cate_list = list(map(match_cate, df[sequence_features[3]].values))
        neg_hist_cate_list = pad_sequences(neg_hist_cate_list, maxlen=sequence_length, padding='post', )

        return hist_item_list, neg_hist_item_list, hist_item_len, hist_cate_list, neg_hist_cate_list, hist_cate_len
    else:
        pass


def scaler_dense_feature(df, dense_featues, scaler_type='log'):
    """
    :param df:
    :param dense_featues:
    :param scaler_type:
    :return:
    """

    def minmaxscaler(row):
        #  需要将min和max写死
        pass

    def ln1p_scaler(row):
        return np.log1p(row)

    def lg1p_scaler(row):
        return np.log10(1 + row)

    if scaler_type is 'log':
        for col in tqdm(dense_featues, desc='scaler dense features'):
            ulimit = df[col].max()
            if ulimit > 1000:
                df[col] = df[col].apply(lg1p_scaler)
            elif 1 < ulimit <= 1000:
                df[col] = df[col].apply(ln1p_scaler)
            else:
                df[col] = df[col]

    return df
