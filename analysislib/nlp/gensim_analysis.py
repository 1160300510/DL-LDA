import pandas as pd
import preprocess.pre as pre
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
# Remove rare and common tokens.
from gensim.corpora import Dictionary
# Train LDA model.
from gensim.models import LdaModel
from pprint import pprint
import numpy as np
import logging
from gensim.models import CoherenceModel
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':

    df = pd.read_csv('gensim.csv')
    docs = []
    for index, row in df.iterrows():
        title = str(row['Title'])
        tags = str(row['Tags'])
        body = str(row['Body'])
        title = pre.processbody(title)
        body = pre.processbody(body)
        tags = pre.preprocesstag(tags)
        doc = title + " " + tags + " " + body
        docs.append(doc)
    print(len(docs))
    print(docs[0][:50])

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]
    np.save('gensim_docs.npy', np.array(docs))
    # docs = np.load('gensim_docs.npy', allow_pickle=True).tolist()

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)
    dictionary.save('gensim.dict')
    dictionary.filter_extremes(no_below=10, no_above=0.8)
    # dictionary = Dictionary.load('gensim.dict')

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    np.save('gensim.npy', np.array(corpus))
    # corpus = np.load('gensim.npy', allow_pickle=True).tolist()
    print('Number of unique tokens: %d' % len(dictionary))
    # print('Number of documents: %d' % len(corpus))
    # %%
    best_coherence = -100
    best_num_topics = 0
    coherences = []
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    # for i in range(5, 15):
    #
    #     num_topics = i
    #     model = LdaModel(
    #         corpus=corpus,
    #         id2word=id2word,
    #         chunksize=chunksize,
    #         alpha='auto',
    #         eta='auto',
    #         iterations=iterations,
    #         num_topics=num_topics,
    #         passes=passes,
    #         eval_every=eval_every
    #     )
    #
    #     top_topics = model.top_topics(corpus)  # , num_words=20)
    #     coherence_model_lda = CoherenceModel(model=model, texts=docs, corpus=corpus, dictionary=dictionary, coherence='c_v')
    #     coherence_lda = coherence_model_lda.get_coherence()
    #
    #     coherences.append(coherence_lda)
    #     if coherence_lda > best_coherence:
    #         best_num_topics = i
    #         best_coherence = coherence_lda
    #         # model.save('../model/nlp_10/nlp_10.model')
    #     topicwords = model.print_topics(num_topics=i, num_words=15)
    #     pprint(topicwords)
    #
    # print("best coherence: " + str(best_coherence))
    # print("best topic nums: " + str(best_num_topics))
    # print(coherences)
    #
    # num_topics = best_num_topics
    # model = LdaModel(
    #     corpus=corpus,
    #     id2word=id2word,
    #     chunksize=chunksize,
    #     alpha='auto',
    #     eta='auto',
    #     iterations=iterations,
    #     num_topics=num_topics,
    #     passes=passes,
    #     eval_every=eval_every
    # )
    # top_topics = model.top_topics(corpus)  # , num_words=20)
    # coherence_model_lda = CoherenceModel(model=model, texts=docs, corpus=corpus, dictionary=dictionary, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # # model.save('../model/nlp_gensim/nlp_gensim.model')
    # topicwords = model.print_topics(num_topics=best_num_topics, num_words=15)
    # pprint(topicwords)
    # print(coherence_lda)
    num_topics = 7
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    top_topics = model.top_topics(corpus)  # , num_words=20)
    coherence_model_lda = CoherenceModel(model=model, texts=docs, corpus=corpus, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    model.save('nlp_gensim_7.model')
    topicwords = model.print_topics(num_topics=num_topics, num_words=15)
    pprint(topicwords)
    print(coherence_lda)
    #
    # num_topics = 10
    # model = LdaModel(
    #     corpus=corpus,
    #     id2word=id2word,
    #     chunksize=chunksize,
    #     alpha='auto',
    #     eta='auto',
    #     iterations=iterations,
    #     num_topics=num_topics,
    #     passes=passes,
    #     eval_every=eval_every
    # )
    # top_topics = model.top_topics(corpus)  # , num_words=20)
    # coherence_model_lda = CoherenceModel(model=model, texts=docs, corpus=corpus, dictionary=dictionary, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # model.save('nlp_gensim_10.model')
    # topicwords = model.print_topics(num_topics=10, num_words=15)
    # pprint(topicwords)
    # print(coherence_lda)
    #
    # num_topics = 15
    # model = LdaModel(
    #     corpus=corpus,
    #     id2word=id2word,
    #     chunksize=chunksize,
    #     alpha='auto',
    #     eta='auto',
    #     iterations=iterations,
    #     num_topics=num_topics,
    #     passes=passes,
    #     eval_every=eval_every
    # )
    # top_topics = model.top_topics(corpus)  # , num_words=20)
    # coherence_model_lda = CoherenceModel(model=model, texts=docs, corpus=corpus, dictionary=dictionary, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # model.save('nlp_gensim_15.model')
    # topicwords = model.print_topics(num_topics=15, num_words=15)
    # pprint(topicwords)
    # print(coherence_lda)
    #
    # num_topics = 20
    # model = LdaModel(
    #     corpus=corpus,
    #     id2word=id2word,
    #     chunksize=chunksize,
    #     alpha='auto',
    #     eta='auto',
    #     iterations=iterations,
    #     num_topics=num_topics,
    #     passes=passes,
    #     eval_every=eval_every
    # )
    # top_topics = model.top_topics(corpus)  # , num_words=20)
    # coherence_model_lda = CoherenceModel(model=model, texts=docs, corpus=corpus, dictionary=dictionary, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # model.save('nlp_gensim_20.model')
    # topicwords = model.print_topics(num_topics=20, num_words=15)
    # pprint(topicwords)
    # print(coherence_lda)
