import pandas as pd
import os

items_info1 = pd.read_csv('../data/taobao_items1', sep='\t', names=["itemID", "class1ID", "classID", "brandID", "price"])
items_info2 = pd.read_csv('../data/taobao_items2', sep='\t', names=["itemID", "class1ID", "classID", "brandID", "price"])

items_info = items_info1.append(items_info2)
items_info = items_info.reset_index(drop='index')
items_info.to_csv('taobao_items', sep='\t', index=False, header=None)

os.system('rm taobao_items1 taobao_items2')