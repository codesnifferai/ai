#!/usr/bin/env bash

for filename in ./data/**/*.java; do    
    encoding=$(file -i "$filename" | sed -n -e 's/^.*charset=//p' | tr a-z A-Z | sed 's/ *$//g')
    if [[ $encoding != "UTF-8" && $encoding != "US-ASCII" ]]; then
        printf "$filename $encoding \n"        
        iconv -f "${encoding}" -t "UTF-8" "$filename" > converted.txt &&
        mv "converted.txt" "$filename"
    fi
done