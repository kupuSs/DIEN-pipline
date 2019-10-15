"""
@author kupuV
@time 20191003
"""
import sys
sys.path.append('../')
from main.dien import DIEN
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.optimizers import SGD, RMSprop, Adam, Adagrad
from tensorflow.python.keras.utils import multi_gpu_model
from sklearn.metrics import roc_auc_score, mean_squared_error
from data_process.get_standard_input import get_standard_input
import os
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

flags = tf.app.flags
# data preprocess params(目前特征仅是简单处理，后续可能会加入很多特征处理的trick)
flags.DEFINE_string('train_path', '../data/amazon_books_train.csv', '训练集路径')
flags.DEFINE_string('test_path', '../data/amazon_books_test.csv', '测试集路径')
flags.DEFINE_boolean('predict_by_chunks', False, '是否按chunks进行操作(处理大规模数据)')
flags.DEFINE_integer('data_type', 1, '测试数据集(1->amazon & 2->taobao & 3->customize)')
flags.DEFINE_string('normalized_mode', 'log', '特征归一化方式(log & minmax)')
flags.DEFINE_integer('sequence_length', 20, 'sequence长度')
flags.DEFINE_boolean('use_parallel', False, '特征预处理部分是否采用并行操作')

# model params
flags.DEFINE_integer('embedding_size', 20, '类别特征embedding size')
flags.DEFINE_string('dnn_hidden_units', '[200, 80]', 'dnn网络结构')
flags.DEFINE_string('att_hidden_units', '[80, 40]', 'attention网络结构')
flags.DEFINE_string('dnn_activation', 'relu', 'dnn层激活函数(relu&dice&etc)')
flags.DEFINE_string('att_activation', 'dice', 'attention层激活函数(relu & dice & etc)')
flags.DEFINE_string('optimizer', 'adam', '定义优化器(sgd, adam, rmsprop, adagrad& etc)')
flags.DEFINE_float('learning_rate', 0.001, '定义学习率')
flags.DEFINE_float('dnn_dropout', 0, 'dropout rate')
flags.DEFINE_string('gru_type', 'AUGRU', 'GRU更新方式(GRU & AGRU & AIGRU & AUGRU)')
flags.DEFINE_float('auxloss_alpha', 1.0, '辅助loss权重')
flags.DEFINE_boolean('use_negsampling', True, '是否对sequence进行负采样')
flags.DEFINE_integer('task_mode', 1, '任务类型(1->binary & 2->regression)')
flags.DEFINE_boolean('use_batch_normalization', False, '是否使用批归一化')
flags.DEFINE_float('l2_reg_dnn', 0, 'dnn层l2正则化系数')
flags.DEFINE_float('l2_reg_embedding', 0, 'embedding层l2正则化系数')
flags.DEFINE_float('init_std', 0.0001, '网络参数初始化标准差(正态分布)')
flags.DEFINE_integer('seed', 2019, '模型种子')

# other params
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_integer('global_epochs', 4, '全局epochs')
flags.DEFINE_integer('inner_epochs', 1, 'epochs')
flags.DEFINE_integer('verbose', 1, 'verbose')
flags.DEFINE_float('lr_decay', 1.0, '学习率衰减系数')
flags.DEFINE_boolean('use_gpu', False, '是否使用gpu')
flags.DEFINE_integer('gpu_nums', 2, '使用gpu的个数')
flags.DEFINE_string('gpu_devices', '0, 1', '指定gpu device id')
flags.DEFINE_boolean('save_model', True, '是否存储模型')

FLAGS = flags.FLAGS
FLAGS.dnn_hidden_units = eval(FLAGS.dnn_hidden_units)
FLAGS.att_hidden_units = eval(FLAGS.att_hidden_units)
if FLAGS.data_type == 1:
    FLAGS.data_type = 'amazon_books'
elif FLAGS.data_type == 2:
    FLAGS.data_type = 'taobao'
elif FLAGS.data_type == 3:
    FLAGS.data_type = 'customize'
else:
    raise Exception('wrong data_type {}, please choose 1(amazon_books), 2(taobao), 3(customize)'.format(FLAGS.data_type))

if FLAGS.task_mode == 1:
    FLAGS.task_mode = 'binary'
elif FLAGS.task_mode == 2:
    FLAGS.task_mode = 'regression'
else:
    raise Exception('wrong task_mode {}, please choose 1(binary) or 2(regression)'.format(FLAGS.task_mode))

feature_columns, behavior_feature_list, train_x, train_y, test_x, test_y = get_standard_input(
    train_path=FLAGS.train_path,
    test_path=FLAGS.test_path,
    predict_by_chunks=FLAGS.predict_by_chunks,
    data_type=FLAGS.data_type,
    scaler_type=FLAGS.normalized_mode,
    sequence_truncated=FLAGS.sequence_length,
    use_parallel=FLAGS.use_parallel  # 小数据集不要使用并行
)

if FLAGS.optimizer == 'sgd':
    optimizer = SGD(lr=FLAGS.learning_rate)
elif FLAGS.optimizer == 'adam':
    optimizer = Adam(lr=FLAGS.learning_rate)
elif FLAGS.optimizer == 'rmsprop':
    optimizer = RMSprop(lr=FLAGS.learning_rate)
elif FLAGS.optimizer == 'adagrad':
    optimizer = Adagrad(lr=FLAGS.learning_rate)
else:
    # 可以定义其他优化器
    pass

print('-' * 10, '> model params:')
params_df = pd.DataFrame(
    data=[
        FLAGS.embedding_size, FLAGS.dnn_hidden_units, FLAGS.att_hidden_units,
        FLAGS.dnn_activation, FLAGS.att_activation, FLAGS.dnn_dropout, FLAGS.gru_type,
        FLAGS.auxloss_alpha, FLAGS.l2_reg_dnn, FLAGS.l2_reg_embedding, FLAGS.init_std,
        FLAGS.seed, FLAGS.use_batch_normalization, FLAGS.use_negsampling, FLAGS.task_mode,
        FLAGS.global_epochs, FLAGS.inner_epochs, FLAGS.batch_size
    ],
    index=[
        'embedding_size', 'dnn_hidden_units', 'att_hidden_units', 'dnn_activation', 'att_activation',
        'dnn_dropout', 'gru_type', 'auxloss_alpha', 'l2_reg_dnn', 'l2_reg_embedding', 'init_std',
        'seed', 'use_batch_normalization', 'use_negsampling', 'task_mode',
        'global_epochs', 'inner_epochs', 'batch_size'
    ],
    columns=['value']
)
print(params_df)

model = DIEN(
    dnn_feature_columns=feature_columns,
    history_feature_list=behavior_feature_list,
    embedding_size=FLAGS.embedding_size,
    dnn_hidden_units=FLAGS.dnn_hidden_units,
    att_hidden_units=FLAGS.att_hidden_units,
    dnn_activation=FLAGS.dnn_activation,
    att_activation=FLAGS.att_activation,
    dnn_dropout=FLAGS.dnn_dropout,
    gru_type=FLAGS.gru_type,
    alpha=FLAGS.auxloss_alpha,
    l2_reg_dnn=FLAGS.l2_reg_dnn,
    l2_reg_embedding=FLAGS.l2_reg_embedding,
    init_std=FLAGS.init_std,
    seed=FLAGS.seed,
    use_bn=FLAGS.use_batch_normalization,
    use_negsampling=FLAGS.use_negsampling,
    task=FLAGS.task_mode
)

if FLAGS.use_gpu and FLAGS.gpu_nums > 0:
    print('-' * 10, '> training with {} gpu-cores'.format(FLAGS.gpu_nums))
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_devices
    model = multi_gpu_model(model, gpus=FLAGS.gpu_num)

if FLAGS.task_mode is 'binary':
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_crossentropy'])
else:
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mse'])

print('-' * 10, '> training with cpu-cores')
for global_epoch in range(FLAGS.global_epochs):
    print('-' * 10, '> global_epoch: ', global_epoch + 1)
    model.fit(
        x=train_x,
        y=train_y,
        validation_data=(test_x, test_y),
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.inner_epochs,
        verbose=FLAGS.verbose,
        callbacks=[
                LearningRateScheduler(lambda epoch: FLAGS.learning_rate * (FLAGS.lr_decay ** global_epoch))
            ]
    )
    tr_pred_ans = model.predict(train_x, batch_size=FLAGS.batch_size)
    te_pred_ans = model.predict(test_x, batch_size=FLAGS.batch_size)
    if FLAGS.task_mode is 'binary':
        print('-' * 10, "> train AUC", round(roc_auc_score(train_y, tr_pred_ans), 4))
        print('-' * 10, "> test AUC", round(roc_auc_score(test_y, te_pred_ans), 4))
    else:
        print('-' * 10, "> train mse", round(mean_squared_error(train_y, tr_pred_ans), 4))
        print('-' * 10, "> test mse", round(mean_squared_error(test_y, te_pred_ans), 4))

if FLAGS.save_model:
    print('-' * 10, '> saving model')
    model.save('../model/model_' + str(datetime.date.today()) + '.h5')