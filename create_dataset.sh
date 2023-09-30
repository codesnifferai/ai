#!/usr/bin/env bash
python3 web_scraping/web_scraping.py
rm -rf -- /tmp/code_sniffer_ai_tmp_*
find data/code_files/ -type f | xargs chmod -x 
./web_scraping/change_encoding.sh
python3 ./web_scraping/remove_license.py