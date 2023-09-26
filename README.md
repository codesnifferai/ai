# Code Sniffer AI - Data and AI repository

This is where the data extraction and AI training will be done. You don't need this repository to run the app, only the parameters file. Access the [main repository](https://github.com/codesnifferai/codesnifferai/) for more information.

## Commit naming convention
Commits should begin with the area in which you are working (name of the subdirectory, eg. model or web_scraping) followed by a brief description of the modification. Then, write a full description of the problem and how you commit solves it. All verbs should be in imperative mode.

All commits should be signed. Add `-s` to the command to do it automatically.

Example:

```
web-scraping: add check to skip private repositories.

Some urls in the dataset belongs to discontinued projects, with private
repositories. Add verification to skip these repositories when scraping.

Signed-off-by: João Gabriel Josephik <joao.gabrielaaj@gmail.com>
```
