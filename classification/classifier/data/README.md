# Dati di input per preprocessing e classificazione

* <b>philosophy.csv</b>: tesi di filosofia (estratte con regex), con e senza abstract, totale: 6404
* <b>no_philosophy.csv</b>: tesi non filosofiche, totale: 316364

I due file .csv vengono prodotti in output dallo scipt <tt>prepare_datasets.py</tt> e messi in questa directory.

Il modello dei file è il seguente:
* title
* creator
* university
* publisher
* year
* abstract
* type
* subject
* id
