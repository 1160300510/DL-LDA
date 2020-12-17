select distinct * from Posts where (Tags like '%<nltk>%' or Tags like '%<scikit-learn>%' or Tags like '%<stanford-nlp>%'
or Tags like '%<spacy>%' or Tags like '%<gensim>%' or Tags like '%<textblob>%' or Tags like '%<polyglot>%'
or Tags like '%<opennlp>%' or Tags like '%<charniak-parser>%' or Tags like '%<pytextrank>%' or Tags like '%<huggingface-transformers>%'
or Tags like '%<rasa-nlu>%' or Tags like '%<dialogflow-es>%' or Tags like '%<huggingface-tokenizers>%' or Tags like '%<johnsnowlabs-spark-nlp>%'
or Tags like '%<flair>%' or Tags like '%<sense2vec>%' or Tags like '%<python-pattern>%')
and CreationDate>'2011-12-31' and CreationDate<'2020-09-01';