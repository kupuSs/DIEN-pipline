"""
@author kupuV
@time 20191003
"""

import pandas as pd
from data_process.utils import get_stats, reduce_mem_usage_sd


def data_loader(path, chunkSize=100000, data_type='amazon_books', verbose=False, return_chunks=False):
    reader = pd.read_csv(path, iterator=True)
    loop = True
    chunks = []
    count = 0
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            # if data_type is 'taobao':
            #     chunk = chunk[chunk.date >= 20190617]  # taobao数据量太大，sequence太长，负采样时间太久，顾只取3天
            chunks.append(chunk)
            count += 1
            # if count == 1:
            #     break
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df = pd.concat(chunks, ignore_index=True)

    if verbose is True:
        print("缩小前,数据情况统计")
        stats = get_stats(df)
        print(stats)
    df = reduce_mem_usage_sd(df, verbose=False)
    if verbose is True:
        print("缩小后,数据情况统计")
        stats = get_stats(df)
        print(stats)

    print('-' * 10, '> {} shape {}'.format(path, df.shape))

    if return_chunks:
        return chunks
    else:
        return df
