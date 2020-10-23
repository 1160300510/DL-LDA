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
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    '''
    5: 0.586
    6: 0.608
    7: 0.556
    8: 0.587
    9: 0.582
    10: 0.557
    11: 0.586
    12: 0.612 * 
    13: 0.564
    14: 0.553
    15: 0.564
    16: 0.587
    17: 0.581
    18: 0.603
    19: 0.557
    20: 0.574
    21: 0.546
    22: 0.562
    23: 0.540
    24: 0.536
    '''

    best_coherence = -100
    best_num_topics = 0
    coherences = []
    dictionary = Dictionary.load('../data/nlp.dict')
    corpus = np.load('../data/nlp_corpus.npy', allow_pickle=True).tolist()
    docs = np.load('../data/nlp_docs.npy', allow_pickle=True).tolist()
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    num_topics = 12
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

    model.save('../model/nlp_12/nlp_12.model')
    model.print_topics(num_topics=num_topics, num_words=15)

    # model = LdaModel.load('../model/nlp_10/nlp_10.model')
    # coherence_model_lda = CoherenceModel(model=model, texts=docs, corpus=corpus, dictionary=dictionary, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)

    # for i in range(5,25):
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
    #     # model.save('../model/nlp_10/nlp_10.model')
    #     model.print_topics(num_topics=i, num_words=15)
    #
    # print("best coherence: "+str(best_coherence))
    # print("best topic nums: "+str(best_num_topics))
    # print(coherences)

    # df = pd.read_csv('../preprocess/nlp-process.csv')
    # docs = []
    # for index, row in df.iterrows():
    #     body = row['Body']
    #     docs.append(body)
    # print(len(docs))
    # print(docs[0][:50])
    #
    # # Split the documents into tokens.
    # tokenizer = RegexpTokenizer(r'\w+')
    # for idx in range(len(docs)):
    #     docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
    #
    # # Remove numbers, but not words that contain numbers.
    # docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    #
    # # Remove words that are only one character.
    # docs = [[token for token in doc if len(token) > 1] for doc in docs]
    #
    # # lemmatizer = WordNetLemmatizer()
    # # docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    # #
    # # # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    # # bigram = Phrases(docs, min_count=20)
    # # for idx in range(len(docs)):
    # #     for token in bigram[docs[idx]]:
    # #         if '_' in token:
    # #             # Token is a bigram, add to document.
    # #             docs[idx].append(token)
    #
    # # Create a dictionary representation of the documents.
    # dictionary = Dictionary(docs)
    # dictionary.save('../data/nlp.dict')
    # # dictionary = Dictionary.load('../data/nlp.dict')
    #
    # # Bag-of-words representation of the documents.
    # corpus = [dictionary.doc2bow(doc) for doc in docs]
    # np.save('../data/nlp_corpus.npy',np.array(corpus))
    # # corpus = np.load('../data/nlp_corpus.npy').tolist()
    #
    #
    # print('Number of unique tokens: %d' % len(dictionary))
    # print('Number of documents: %d' % len(corpus))
    #
    # # for i in range(10,30):
    # #     trainmodel(i, corpus, dictionary.id2token)
    #
    # num_topics = 10
    # chunksize = 2000
    # passes = 30
    # iterations = 400
    # eval_every = None  # Don't evaluate model perplexity, takes too much time.
    #
    # # # Make a index to word dictionary.
    # temp = dictionary[0]  # This is only to "load" the dictionary.
    # id2word = dictionary.id2token
    #
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
    #
    # top_topics = model.top_topics(corpus)  # , num_words=20)
    #
    # # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    # avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    # print('Average topic coherence: %.4f.' % avg_topic_coherence)
    # pprint(top_topics)
    # model.save('../model/nlp_10/nlp_10.model')
    # model.print_topics(10,15)









