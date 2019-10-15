# DIEN-PIPLINE
DIEN-pipline implement
一个DIEN的pipline简单实现，包括以下部分：
* 模型本身
* 数据预处理
* 负采样实现
* 简易搭建，只需要根据数据填写config文件即可

该pipline实现了数据获取，数据预处理，sequence负采样的集成，且是解耦的，可以快速应用于新的数据集，实现了amazon和taobao两个数据集的baseline


## 感谢以下大神的开源代码
* code: https://github.com/shenweichen/DeepCTR
* author: shenweichen

* code: https://github.com/mouna99/dien
* author: alibaba


## 参考论文
>Zhou G, Mou N, Fan Y, et al. Deep interest evolution network for click-through rate prediction[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 5941-5948.[Paper in arXiv] (https://arxiv.org/abs/1809.03672).


## 环境配置
* python 3.6.5
* tensorflow == 1.13.1
* numpy == 1.17.2 
* sklearn == 0.20.3


## Dataset
* 提供已经预处理过的数据集: Amazon-book, TaoBao
* 原始数据集 [Amazon-book] (http://jmcauley.ucsd.edu/data/amazon), [TaoBao] (https://tianchi.aliyun.com/competition/entrance/231719/information)
---
* [Amazon-book]
* data/amazon_books_train.csv.zip
* data/amazon_books_test.csv.zip

* [TaoBao]
* data/taobao_train.csv.zip
* data/taobao_test.csv.zip
* data/taobao_items1.zip
* data/taobao_items2.zip
---
* 注意: taobao_items1.zip与taobao_items2.zip需要拼接
```
cd data
python concat_taobao_items.py
```


## baseline示例：
```
cd run
```
* amazon_books dataset
```
python run.py --train_path ../data/amazon_books_train.csv --test_path ../data/amazon_books_test.csv --data_type 1 --sequence_length 50 --embedding_size 20 --dnn_hidden_units '[200, 80]' --att_hidden_units '[80, 40]' --dnn_activation relu --att_activation dice --optimizer adam --learning_rate 0.001 --dnn_dropout 0 --auxloss_alpha 1.0 --seed 2019 --task_mode 1 --batch_size 128 --global_epochs 4 --inner_epochs 1 --verbose 1 --lr_decay 1.0
```

* TaoBao dataset
```
python run.py --train_path ../data/taobao_train.csv --test_path ../data/taobao_test.csv --data_type 2 --sequence_length 20 --embedding_size 20 --dnn_hidden_units '[200, 80]' --att_hidden_units '[80, 40]' --dnn_activation relu --att_activation dice --optimizer adam --learning_rate 0.001 --dnn_dropout 0 --auxloss_alpha 1.0 --seed 2019 --task_mode 1 --batch_size 128 --global_epochs 4 --inner_epochs 1 --verbose 1 --lr_decay 1.0
```


## 快速构建其它数据集的baseline
```
cd data_process
vim config_customize.py
```
打开config_customize.py后，你会看到这些:
```
sparse_features = []  # 类别特征

dense_features = []  # 统计特征

behavior_feature_list = []  # sequence特征对应的query特征

sequence_features = []  # sequence特征

target = []  # label
```
只需要将相应特征名称填写进去即可
可以参考amazon和taobao的config文件
```
# amazon_books
sparse_features = ['userID', 'itemID', 'cateID']

dense_features = []

behavior_feature_list = ["itemID", "cateID"]

sequence_features = [
    'hist_item_list',
    'hist_cate_list'
]

target = ['label']


# taobao
sparse_features = [
    'userID', 'itemID', 'class1ID', 'classID', 'brandID', 'gender', 'age_level', 'education_degree', 'career_type']

multi_hot_features = ['stage']

dense_features = ['price', 'income']

behavior_feature_list = ["itemID", "class1ID"]

sequence_features = ['hist_item_list', 'hist_class_list']

target = ['behavior']
```
这里有2点需要注意
* sequence_features需要是协同的，比如amazon数据集中，item的类别是cate，所以hist_cate_list算是协助hist_item_list的特征，自己构建时也要按照这个标准
* sequence_features就是behavior_feature_list中特征的sequence，需要注意

配置好config文件后，需要在进行negative sampling时指定behavior_feature_list中的特征，具体操作如下:
```
cd data_process
vim get_standard_input.py
```
代码定位到line 28，你会看到
```
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
```
按要求填写behavior_features的特征名称即可

以上便是搭建其它数据集baseline的方法，具体细节可以参考amazon和taobao数据集baseline的实现


## 一些重要的参数:
* `predict_by_chunks`
  * 是否按chunks进行操作(处理大规模数据)，如果使用，需要做简单的二次开发. 用法 `--predict_by_chunks True`.

* `normalized_mode`
  * 特征归一化方式（log & minmax）. 用法 `--normalized_mode log`.
  * minmax方式要考虑到数据长尾分布的情况，后续会实现.

* `use_parallel`
  * 特征预处理部分是否采用并行操作. 用法 `--use_parallel True`.
  * 待实现.
    
* `use_gpu`
  * 是否使用gpu. 用法 `--use_gpu True`.

* `gpu_nums`
  * 是否使用gpu. 用法 `--gpu_nums 2`.
  * 如果使用多gpu，则模型会在数据处理部分并行化

* `gpu_devices`
  * 指定gpu device id. 用法 `--gpu_devices 0, 1`.

* `save_model`
  * 是否存储模型. 用法 `--save_model False`.

