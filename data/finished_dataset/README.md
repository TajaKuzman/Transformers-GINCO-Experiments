# GINCO - Genre IdeNtification COrpus

## Files

This dataset consists of two files: `suitable.json` and `nonsuitable.json`.

## File structure

Each file is a json packaged list of documents. Documents in the `suitable.json` file have the following fields:

| field             | description                                                    | remarks                                                                                                            |
|-------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| id                | document unique string id                                      |                                                                                                                    |
| url               | url of the original website from which the text was scrapped   |                                                                                                                    |
| crawled           | string, crawl year                                             |                                                                                                                    |
| hard              | human annotated, whether the document was hard to label        |                                                                                                                    |
| paragraphs        | list of dictionaries, containing paragraphs and their metadata |                                                                                                                    |
| primary_level_N   | human annotated primary label                                  | level indicates label downsampling extent, 1 being not at all and 3 being downcast from 25 labels to 12            |
| secondary_level_N | human annotated secondary label                                |                                                                                                                    |
| tertiary_level_N  | human annotated primary label                                  |                                                                                                                    |
| split             | whether the document belongs to train, dev, or test split      | 60:20:20 split, stratified by primary_level_2, with documents with the same domain strictly kept in the same split |
| domain            | domain from which the document was scrapped                    | parsed from url field                                                                                              |