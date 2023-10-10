#!/usr/bin/env python3
import os
import re
import sys

if len(sys.argv) < 2:
    print(f"No dataset path provided. \nUsage: {sys.argv[0]} DATASET_PATH")
    exit(1)
DATASET_DIR = sys.argv[1]

for filename in os.listdir(DATASET_DIR):
    try:        
        f = os.path.join(DATASET_DIR, filename)
        # checking if it is a file
        print(f)
        if os.path.isfile(f):
            # print(f)
            with open(f, "r") as file:
                string = file.read()
            count_comment = 0            
            start = string[0:2]
            if start != '/*':
                print(f"Skipping {filename}. It does not contain a license.")
                continue
            end = 2
            while string[end-2:end] != "*/" and end < len(string):
                end += 1
            if end==len(string):
                print("ERROR: Ending of comment block not found")
            
            string = string[end:]                   
            print(end)           
            
            with open(f, "w") as file:
                file.write(string)
    except Exception as e:
        print(e)
        print(f"Removing {f}.")
        os.remove(f)


            


            

