#!/usr/bin/env python3
import pandas as pd
import os
old_csv_df = pd.read_csv('data/data.csv')

old_csv_df.set_index('Address', inplace=True)
code_files = os.listdir('data/code_files')


class_list = []

for file in code_files:
    classname = file.replace('.java', '')
    class_list.append(classname)
columns_filter=['Brain Class',
                'Data Class',
                'Futile Abstract Pipeline',
                'Futile Hierarchy',
                'God Class',
                'Hierarchy Duplication',
                'Model Class',
                'Schizofrenic Class']
new_csv_df = old_csv_df[old_csv_df.index.isin(class_list)][columns_filter]

new_csv_df.index = new_csv_df.index.astype(str) + '.java'
new_csv_df.to_csv('data/filtered_data.csv')