# GINCO - Genre IdeNtification COrpus

## Files

This dataset consists of two files: `suitable.json` and `nonsuitable.json`.

## File structure

Each file is a json packaged list of documents. Documents in the `suitable.json` file have the following fields:

| field             | description                                                    | remarks                                                                                                            |
|-------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `id`                | document unique string id                                      |                                                                                                                    |
| `url`               | url of the original website from which the text was scrapped   |                                                                                                                    |
| `crawled`           | string, crawl year                                             |                                                                                                                    |
| `hard`              | human annotated, whether the document was hard to label        |                                                                                                                    |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata |                                                                                                                    |
| `primary_level_N`   | human annotated primary label                                  | `N` ∈ {1,2,3} indicates label downsampling extent, 1 being not at all and 3 being downcast from 25 labels to 12    |
| `secondary_level_N` | human annotated secondary label                                |                                                                                                                    |
| `tertiary_level_N`  | human annotated primary label                                  |                                                                                                                    |
| `split`             | whether the document belongs to train, dev, or test split      | 60:20:20 split, stratified by primary_level_2, with documents with the same domain strictly kept in the same split |
| `domain`            | domain from which the document was scrapped                    | parsed from url field                                                                                              |



Documents in the `nonsuitable.json` file have the following fields:

| field             | description                                                    | remarks                                                                                                            |
|-------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `id`                | document unique string id                                      |                                                                                                                    |
| `url`               | url of the original website from which the text was scrapped   |                                                                                                                    |
| `crawled`           | string, crawl year                                             |                                                                                                                    |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata |                                                                                                                    |
| `primary_level_1`   | human annotated primary label                                  | there was no downcasting for unsuitable dataset|
| `secondary_level_1` | human annotated secondary label                                |                                                                                                                    |
| `split`             | whether the document belongs to train, dev, or test split      | 60:20:20 split, stratified by primary_level_1|
| `domain`            | domain from which the document was scrapped                    | parsed from url field                                                                                              |


## Paragraph structure

Items of the list in `paragraphs` have the following fields:

| field     | description                                                              | remarks                             |
|-----------|--------------------------------------------------------------------------|-------------------------------------|
| `text`      | paragraph text                                                           |                                     |
| `duplicate` | boolean, result of automated deduplication                               |                                     |
| `keep`      | boolean, human annotated tag whether or not the paragraph should be kept | only present in suitable paragraphs |