#!/usr/bin/env python3
import os
import re
import sys
from tqdm import tqdm

if len(sys.argv) < 2:
    print(f"No dataset path provided. \nUsage: {sys.argv[0]} DATASET_PATH")
    exit(1)
DATASET_DIR = sys.argv[1]

for filename in tqdm(os.listdir(DATASET_DIR)):
    try:        
        f = os.path.join(DATASET_DIR, filename)       
        if os.path.isfile(f):            
            with open(f, "r") as file:
                string = file.read()
            count_comment = 0            
            start = string[0:2]
            if start != '/*':                
                continue
            end = 2
            while string[end-2:end] != "*/" and end < len(string):
                end += 1
            string = string[end:]                   
       
            
            with open(f, "w") as file:
                file.write(string)
    except Exception as e:
        print(e)
        os.remove(f)


            


            

