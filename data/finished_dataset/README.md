# GINCO - Genre IdeNtification COrpus

## Files

This dataset consists of two files: `suitable.json` and `nonsuitable.json`.

## File structure

Each file is a json packaged list of documents. Documents in the `suitable.json` file have the following fields:

| field               | description                                                             | remarks                                                                                                            |
| ------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `id`                | string, document unique id                                              |                                                                                                                    |
| `url`               | string, url of the original website from which the text was scrapped    |                                                                                                                    |
| `crawled`           | string, crawl year                                                      |                                                                                                                    |
| `hard`              | boolean, human annotated, whether the document was hard to label        |                                                                                                                    |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata          |                                                                                                                    |
| `primary_level_N`   | string, human annotated primary label                                   | `N` ∈ {1,2,3} indicates label downsampling extent, 1 being not at all and 3 being downcast from 24 labels to 12    |
| `secondary_level_N` | string, human annotated secondary label                                 | see above                                                                                                          |
| `tertiary_level_N`  | string, human annotated primary label                                   | see above                                                                                                          |
| `split`             | string, whether the document belongs to *train*, *dev*, or *test* split | 60:20:20 split, stratified by primary_level_2, with documents with the same domain strictly kept in the same split |
| `domain`            | string, domain from which the document was scrapped                     | parsed from `url` field                                                                                            |



Documents in the `nonsuitable.json` file have the following fields:

| field               | description                                                          | remarks                                         |
| ------------------- | -------------------------------------------------------------------- | ----------------------------------------------- |
| `id`                | string, document unique string id                                    |                                                 |
| `url`               | string, url of the original website from which the text was scrapped |                                                 |
| `crawled`           | string, crawl year                                                   |                                                 |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata       |                                                 |
| `primary_level_1`   | string, human annotated primary label                                | there was no downcasting for unsuitable dataset |
| `secondary_level_1` | string, human annotated secondary label                              |                                                 |
| `split`             | string, whether the document belongs to train, dev, or test split    | 60:20:20 split, stratified by primary_level_1   |
| `domain`            | string, domain from which the document was scrapped                  | parsed from `url` field                         |


## Paragraph structure

Items of the list in `paragraphs` have the following fields:

| field       | description                                                                             | remarks                             |
| ----------- | --------------------------------------------------------------------------------------- | ----------------------------------- |
| `text`      | string, paragraph text                                                                  |                                     |
| `duplicate` | boolean, result of automated deduplication with [Onion](http://corpus.tools/wiki/Onion) |                                     |
| `keep`      | boolean, human annotated tag whether or not the paragraph should be kept                | only present in suitable paragraphs |

## Sample data instance

```
{'id': '3949',
 'url': 'http://www.pomurje.si/aktualno/sport/zimska-liga-malega-nogometa/',
 'crawled': '2014',
 'hard': False,
 'paragraphs': [{'text': 'Šport', 'duplicate': False, 'keep': True},
  {'text': 'Zimska liga malega nogometa sobota, 12.02.2011',
   'duplicate': False,
   'keep': True},
  {'text': 'avtor: Tonček Gider', 'duplicate': False, 'keep': True},
  {'text': "V 7. krogu zimske lige v malem nogometu v Križevcih pri Ljutomeru je v prvi ligi vodilni 100 plus iz Križevec izgubil s tretjo ekipo na lestvici Rock'n roll iz Križevec z rezultatom 1:2, druga na lestvici Top Finedika iz Križevec je bila poražena z ekipo Bar Milene iz Ključarovec z rezultatom 7:8. V drugi križevski ligi je vodilni Cafe del Mar iz Vučje vasi premagal Montažo Vrbnjak iz Stare Nove vasi z rezultatom 3:2.",
   'duplicate': False,
   'keep': True},
  {'text': 'oglasno sporočilo', 'duplicate': False, 'keep': True},
  {'text': 'Ocena', 'duplicate': False, 'keep': True},
  {'text': 'Komentiraj Za komentiranje ali ocenjevanje moraš biti registriran in prijavljen uporabnik. Registriraj se!',
   'duplicate': True,
   'keep': False}],
 'primary_level_1': 'News/Reporting',
 'primary_level_2': 'News/Reporting',
 'primary_level_3': 'News/Reporting',
 'secondary_level_1': '',
 'secondary_level_2': '',
 'secondary_level_3': '',
 'tertiary_level_1': '',
 'tertiary_level_2': '',
 'tertiary_level_3': '',
 'split': 'test',
 'domain': 'www.pomurje.si',
}
 ```