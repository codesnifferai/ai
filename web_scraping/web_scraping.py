#!/usr/bin/env python3
import util
import pandas as pd
import os
from glob import glob
import uuid
import shutil

def git_clone(repo_url, target_dir):
    cmd = f"git clone {repo_url} {target_dir}"
    os.popen(cmd).read()

def git_delete(repo_dir):
    os.system(f'rm -rf -- "{repo_dir}"')

df = pd.read_csv('./data/urls.csv')

rnd = uuid.uuid4().hex
path = f'/tmp/code_sniffer_ai_tmp_{rnd}'

cn_list = util.create_search_list()

print(cn_list[0:15])
if not os.path.exists('data/code_files'):
    os.mkdir('data/code_files')
    
for index, row in df.iterrows():
    url = row['link']    
    git_clone(url, path)
    for file in glob(f"{path}/**/*.java", recursive=True):
        classname = os.path.basename(file).split('.')[0]
        ind = util.bin_search(cn_list, classname)
        if (ind != -1):
            dir_str = file.split('/')
            class_str = cn_list[ind][1].split('.')
            match = True
            
            # print(dir_str)
            # print(class_str)
            for i in range(2, len(class_str)+1):
                # print(f"Comparing {dir_str[-i]} and {class_str[-i]}")
                if (dir_str[-i] != class_str[-i]):
                    match = False
                    break
            if match:
                print(f"CLASS: {classname} IND: {ind}")
                shutil.copy(file, f"data/code_files/{cn_list[ind][1]}.java")

    git_delete(path)



