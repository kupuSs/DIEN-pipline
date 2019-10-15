"""
@author kupuV
@time 20191003
"""

from main.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_fixlen_feature_names, get_varlen_feature_names
from data_process.data_loader import data_loader
from data_process.feature_process import encoder_sparse_feature_train, encoder_sparse_feature_test, negative_sampling, \
    encoder_sequence, scaler_dense_feature, truncated_sequence
import time
import json
import numpy as np
from data_process import config_amazon_books, config_taobao, config_customize


def get_standard_input(train_path, test_path, predict_by_chunks=False, data_type='amazon_books',
                       scaler_type='log', sequence_truncated=20, use_parallel=True):
    """
    :param sequence_truncated: sequence截断长度
    :param scaler_type: log % minmax & etc
    :param predict_by_chunks: 如果处理大规模数据，需要分块
    :param use_parallel:
    :param train_path:
    :param test_path:
    :param data_type: amazon_books & taobao & etc
    :return:
    """
    global behavior_features1, behavior_features2
    if data_type is 'amazon_books':
        config = config_amazon_books
        behavior_features1 = 'itemID'
        behavior_features2 = 'cateID'
    elif data_type is 'taobao':
        config = config_taobao
        behavior_features1 = 'itemID'
        behavior_features2 = 'classID'
    else:
        config = config_customize
        behavior_features1 = ''  # 这里要定义config.behavior_list中的特征
        behavior_features2 = ''  # 这里要定义config.behavior_list中的特征

    print('-' * 10, '> loading train_set')
    start_time = time.time()
    train = data_loader(train_path, data_type=data_type, return_chunks=predict_by_chunks)
    print('-' * 10, '> loading test_set')
    test = data_loader(test_path, data_type=data_type, return_chunks=predict_by_chunks)
    print("-" * 10, "> loading data cost time: {} minute".format((time.time() - start_time) / 60))

    print('-' * 10, '> sequence_truncated, limited {}'.format(sequence_truncated))
    start_time = time.time()
    train = truncated_sequence(train, config.sequence_features, limit=sequence_truncated)
    test = truncated_sequence(test, config.sequence_features, limit=sequence_truncated)
    print("-" * 10, "> sequence_truncated cost time: {} minute".format((time.time() - start_time) / 60))

    if data_type is 'taobao':
        label_map = {'clk': 0, 'collect': 0, 'cart': 0, 'buy': 1}
        train[config.target[0]] = train[config.target[0]].map(label_map)
        test[config.target[0]] = test[config.target[0]].map(label_map)

    if not use_parallel:

        print('-' * 10, '> encoder_sparse_feature')
        start_time = time.time()
        train = encoder_sparse_feature_train(train, config.sparse_features)
        test = encoder_sparse_feature_test(test, config.sparse_features)
        print("-" * 10, "> encoder sparse feature cost time: {} minutes ---".format((time.time() - start_time) / 60))

        print('-' * 10, '> negative sampling')
        start_time = time.time()
        train, sequence_features = negative_sampling(
            train,
            config.sequence_features,
            features_type=data_type,
            behavior_features1=behavior_features1,
            behavior_features2=behavior_features2,
            is_training=True)
        print("-" * 10, "> train_set negative sampling cost time: {} minutes ---".format((time.time() - start_time) / 60))
        start_time = time.time()
        test, sequence_features = negative_sampling(
            test,
            config.sequence_features,
            features_type=data_type,
            behavior_features1=behavior_features1,
            behavior_features2=behavior_features2,
            is_training=False)
        print("-" * 10, "> test_set negative sampling cost time: {} minutes ---".format((time.time() - start_time) / 60))

        print('-' * 10, '> scaler dense features')
        start_time = time.time()
        train = scaler_dense_feature(train, config.dense_features, scaler_type=scaler_type)
        test = scaler_dense_feature(test, config.dense_features, scaler_type=scaler_type)
        print("-" * 10, "> scaler dense features cost time: {} minutes ---".format((time.time() - start_time) / 60))

        print('-' * 10, '> encoder_sequence')
        start_time = time.time()
        train_hist_item_list, \
        train_neg_hist_item_list, \
        train_hist_item_len, \
        train_hist_cate_list, \
        train_neg_hist_cate_list, \
        train_hist_cate_len = encoder_sequence(train, sequence_features, features_type=data_type, sequence_length=sequence_truncated)

        test_hist_item_list, \
        test_neg_hist_item_list, \
        test_hist_item_len, \
        test_hist_cate_list, \
        test_neg_hist_cate_list, \
        test_hist_cate_len = encoder_sequence(test, sequence_features, features_type=data_type, sequence_length=sequence_truncated)
        print("-" * 10, "> encoder_sequence cost time: {} minutes ---".format((time.time() - start_time) / 60))
    else:
        # 并行模块后续会完成
        pass

    print('-' * 10, '> construct input')

    fixlen_feature_columns = [SparseFeat(feat, train[feat].nunique() + 1) for feat in config.sparse_features]
    fixlen_feature_columns += [DenseFeat(feat, 1,) for feat in config.dense_features]

    with open('../data_cache/' + config.behavior_feature_list[0] + '.json', 'r') as f:
        seq1_2id = json.load(f)

    with open('../data_cache/' + config.behavior_feature_list[1] + '.json', 'r') as f:
        seq2_2id = json.load(f)

    varlen_feature_columns = [
        VarLenSparseFeat('hist_' + config.behavior_feature_list[0], len(seq1_2id) + 1,
                         sequence_truncated, 'mean', embedding_name=config.behavior_feature_list[0]),
        VarLenSparseFeat('hist_' + config.behavior_feature_list[1], len(seq2_2id) + 1,
                         sequence_truncated, 'mean', embedding_name=config.behavior_feature_list[1])
    ]

    feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_columns += [
        VarLenSparseFeat('neg_hist_' + config.behavior_feature_list[0], len(seq1_2id) + 1,
                         sequence_truncated, embedding_name=config.behavior_feature_list[0]),
        VarLenSparseFeat('neg_hist_' + config.behavior_feature_list[1], len(seq2_2id) + 1,
                         sequence_truncated, embedding_name=config.behavior_feature_list[1])
    ]

    fixlen_feature_names = get_fixlen_feature_names(feature_columns)

    train_behavior_length = np.array(train_hist_item_len)
    test_behavior_length = np.array(test_hist_item_len)

    train_model_input = [train[name].values for name in fixlen_feature_names] + \
                        [train_hist_item_list] + \
                        [train_hist_cate_list] + \
                        [train_neg_hist_item_list] + \
                        [train_neg_hist_cate_list] + \
                        [train_behavior_length]
    test_model_input = [test[name].values for name in fixlen_feature_names] + \
                       [test_hist_item_list] + [test_hist_cate_list] + \
                       [test_neg_hist_item_list] + \
                       [test_neg_hist_cate_list] + \
                       [test_behavior_length]

    return feature_columns, config.behavior_feature_list, train_model_input, train[config.target].values, test_model_input, test[config.target].values