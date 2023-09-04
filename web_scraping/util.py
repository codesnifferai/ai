#!/usr/bin/env python3

import pandas as pd
import os
from glob import glob
import pickle


def create_search_list(csv_path: str = './data/data.csv'):
    df = pd.read_csv(csv_path, low_memory=False)
    if os.path.exists('/tmp/code_sniffer_ai_cn_list.pickle'):
        with open('/tmp/code_sniffer_ai_cn_list.pickle', 'rb') as f:
            return pickle.load(f)

    cn_list=[]
    for index, row in df.iterrows():
        cn = str(row['Address'])

        cn_list.append((cn.split('.')[-1], cn)) # (ultimo nome, nome completo da classe)


    cn_list = sorted(cn_list)

    with open('/tmp/code_sniffer_ai_cn_list.pickle', 'wb') as f:
        pickle.dump(cn_list, f)
    
    return cn_list


def bin_search(list: list, item: str) -> int:
    lo = 0
    hi = len(list) - 1
    while lo <= hi:
        mid = (hi + lo) // 2
        
        if (list[mid][0] < item):
            lo = mid + 1
        elif (list[mid][0] > item):
            hi = mid - 1
        else:
            return  mid
    
    return -1

