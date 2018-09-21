# Dati di input per preprocessing e classificazione

* <b>UK_abs_id.csv</b>: tesi di filosofia (estratte con regex), con e senza abstract, totale: 6404
* <b>UK_no_abs_id.csv</b>: tesi non filosofiche, totale: 316364

I due file .csv vengono prodotti in output dallo scipt <tt>prepare_datasets.py</tt> e messi in questa directory.

Il modello dei file è il seguente:
* id
* titolo
* autore
* univ
* publisher
* anno
* abs <b>N.B.</b> Presente solo nel file <i>UK_abs_id.csv</i>
* tipo
* argomento
* preprocessed_data
