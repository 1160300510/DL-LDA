select distinct * from Posts where Id<=63760928 and
(Tags like '%<nlp>%' or Tags like '%<sentiment-analysis>%' or Tags like '%<stanford-nlp>%' or Tags like '%<nltk>%'
or Tags like '%<spacy>%' or Tags like '%<bert-language-model>%' or Tags like '%<opennlp>%' or Tags like '%<ner>%'
or Tags like '%<word-embedding>%' or Tags like '%<n-gram>%' or Tags like '%<text-mining>%' or Tags like '%<named-entity-recognition>%'
or Tags like '%<word2vec>%' or Tags like '%<gensim>%' or Tags like '%<tf-idf>%' or Tags like '%<text-classification>%'
or Tags like '%<lda>%' or Tags like '%<topic-modeling>%' or Tags like '%<pos-tagger>%' or Tags like '%<huggingface-transformers>%'
or Tags like '%<wordnet>%')
and CreationDate>'2012-01-01';