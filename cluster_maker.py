"""cluster_maker.py - contains clustering algorithms that operate on lists of
particular kinds of MediacloudData (Sentences, Paragraphs, or Articles)"""

import json
import os

def all_data_cluster(records):
    """creates 'all_data.txt' file containing all data"""    
    f = open(os.path.dirname(__file__) + "/results/all_clusters.txt",'w')
    all_clusters = []
    for num in range(len(records)):
        all_clusters.append(records[num].__dict__)     
    json.dump(all_clusters,f)
    f.close()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np


def kmeans(records, num_clusters, stopwords=None, features='hashing'):
    """
    records is a list of MediacloudData (Sentences, Paragraphs or Articles)
    
    num_clusters is the number of clusters
    
    stopwords can be 
    (1)'english' 
    (2) a list of strings
    (3) or None (default)
    
    features determines what features will be and can be set to:
    (1) hashing - 0 or 1 if the word is present (default)
    (2) tfidf - tf-idf weighted wordcount
    (3) count - basic wordcount
    
    Default values are set to what has worked best for me. Kmeans takes some time 
    and it may run for several iterations (25/30). To speed it up, decrease the number
    of records that you are clustering over or have lots of patience.
    
    This function prints a Sillhouste score (Determines how good the clustering is)
    """
    
    
    #make corpus (list of strings) for classifier to consume
    corpus = []
    for record in records:
        corpus.append(record.text.lower())
        
    #prep vectorizer
    def get_vectorizer(x):
        return {'hashing': HashingVectorizer(stop_words=stopwords),
        'tfidf': TfidfVectorizer(stop_words=stopwords),
        'count': CountVectorizer(stop_words=stopwords)}[x]
        
    vectorizer = get_vectorizer(features)
    vectors = vectorizer.fit_transform(corpus)
    X = vectors
    true_k = num_clusters
    
    #do clustering
    km = KMeans(n_clusters=true_k, init='k-means++', n_init=1, verbose=1)
    print "Clustering sparse data with %s" % km
    t0 = time()
    km.fit(X)
    print "done in %0.3fs" % (time() - t0)
    
    #Print clustering score
    print "Sillhouste Score = " + str(metrics.silhouette_score(X, km.labels_))
    
    #UNCOMMENT TO SEE SCORE FOR EACH CLUSTER
    #silhouette_samples = metrics.silhouette_samples(X,km.labels_)
    
    #output clusters to file
    for n in range(true_k):
        cluster_content = []
        clustern = np.where(km.labels_ == n)    
        f = open(os.path.dirname(__file__) + "/results/cluster"+str(n)+".txt",'w')
        for num in clustern[0]:
            cluster_content.append(records[num].__dict__)     
        print len(cluster_content) #debugging
        json.dump(cluster_content,f)
        f.close()
        
        
###########################GENSIM STUFF###############################################
# -*- coding: utf-8 -*-
import logging
from pprint import pprint
from gensim import corpora, models, similarities
from nltk.corpus import stopwords 
import string
import math


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def make_documents(texts):
    # remove common words, remove punctuation, and tokenize
    stoplist = stopwords.words('english')
    documents = []
    for text in texts:
        #remove puncutation
        text = text.translate(string.maketrans("",""), string.punctuation)
        #represent text as list of words, remove stopwords
        document = [word for word in text.lower().split() if word not in stoplist]
        #add to documents list
        documents.append(document)
        
    return documents
        
def build_dictionary(documents):
    dictionary = corpora.Dictionary(documents)
    dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference (ask why file looks like that)
    return dictionary
    
def load_dictionary():
    return corpora.Dictionary.load('/tmp/deerwester.dict')
    
class MyCorpus(object):        
        
    def make_document(self, text, remove_stopwords = False):
        # remove common words, remove punctuation, and tokenize
        stoplist = stopwords.words('english')
        #remove puncutation
        text = text.translate(string.maketrans("",""), string.punctuation)
        
        #represent text as list of words, remove stopwords if argument says so
        if not remove_stopwords:
            document = [word for word in text.lower().split()]
        else:
            document = [word for word in text.lower().split() if word not in stoplist]
       
        return document
        
    def __init__(self, sentences):
        self.sentences = sentences
        print "done with init"
        
    def __iter__(self):
        self.dictionary = build_dictionary([])
        for sentence in self.sentences:
            #print dictionary.token2id
            yield self.dictionary.doc2bow(self.make_document(sentence.text, True), allow_update = True)
        
        
## DO LSI    
def latent_semantic_analysis(MediacloudData, num_clusters):   
    corpus = MyCorpus(MediacloudData)#memory friendly implementation
    print "made corpus memory friendly object"
    
    
    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus) #save
    #corpus = corpora.MmCorpus('/tmp/corpus.mm')#load
    print "Serialized corpus"
    
    
    tfidf = models.TfidfModel(corpus) #step 1 -- initialize model
    corpus_tfidf = tfidf[corpus] #step 2 -- use the model to transform vectors
    print "Done with TfIdf"
    #for doc in corpus_tfidf:
    #    print doc
    
    lsi = models.LsiModel(corpus_tfidf, id2word=corpus.dictionary, num_topics=num_clusters)
    corpus_lsi = lsi[corpus_tfidf]
    print "done with lsi"
    
    print lsi.print_topics(9)
    
    #PRINT FOR LSI
    document_topics = []
    for i,doc in enumerate(corpus_lsi):# both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
        selected_topic = 0
        similarity_to_topic = doc[selected_topic][1]
        for topic_number in range(len(doc)):
            if math.fabs(doc[topic_number][1]) > math.fabs(doc[selected_topic][1]):
                selected_topic = topic_number
                similarity_to_topic = math.fabs(doc[topic_number][1])
        document_topics.append(selected_topic)
        MediacloudData[i].topic = selected_topic
        MediacloudData[i].similarity_to_topic = similarity_to_topic
    
    
    import pylab as pl
    pl.hist(document_topics, bins=num_clusters)
    pl.show() 
    
    topic_content = []
    for i in range (num_clusters):
        topic_content.append([])
        
    for i in range(len(MediacloudData)):
        topic_content[document_topics[i]].append(MediacloudData[i].__dict__)
    
    import json
    n=0
    for content in topic_content:
        f = open(os.path.dirname(__file__) + "/results/cluster"+str(n)+".json",'w')
        json.dump(sorted(content,  key=lambda sentence: sentence["similarity_to_topic"], reverse = True),f)
        f.close()
        n=n+1
        
    
    ########################################
    ############## HISTOGRAMS ##############
    ########################################
    #import numpy as np
    #import pylab as pl
    #for my_list in topics_4:
    #    pl.hist(my_list, alpha = .2)
    #    pl.show()
    ########################################

def latent_drichlet_allocation(MediacloudData, num_clusters):
    ## DO LSI       
    corpus = MyCorpus(MediacloudData)#memory friendly implementation
    print "made corpus memory friendly object"
    
    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus) #save
    #corpus = corpora.MmCorpus('/tmp/corpus.mm')#load
    print "Serialized corpus"
    
    
    tfidf = models.TfidfModel(corpus) #step 1 -- initialize model
    corpus_tfidf = tfidf[corpus] #step 2 -- use the model to transform vectors
    print "Done with TfIdf"
    #for doc in corpus_tfidf:
    #    print doc
    
    from gensim.models import hdpmodel, ldamodel
    num_clusters = 9
    lda = ldamodel.LdaModel(corpus_tfidf, id2word=corpus.dictionary, num_topics=num_clusters)
    corpus_lda = lda[corpus]
    print "done with LDA"
    print lda.print_topics(num_clusters)
    
    document_topics = []
    for i,doc in enumerate(corpus_lda):# both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
        selected_topic = 0
        similarity_to_topic = doc[selected_topic][1]
        for topic_number in range(len(doc)):
            if math.fabs(doc[topic_number][1]) > math.fabs(doc[selected_topic][1]):
                selected_topic = topic_number
                similarity_to_topic = math.fabs(doc[topic_number][1])
        document_topics.append(selected_topic)
        MediacloudData[i].topic = selected_topic
        MediacloudData[i].similarity_to_topic = similarity_to_topic
    
    import pylab as pl
    pl.hist(document_topics, bins=num_clusters)
    pl.title('Distribution of Data in Topics')
    pl.xlabel('Topic #')
    pl.ylabel('#of Data Points')
    pl.show() 
    
    topic_content = []
    for i in range (num_clusters):
        topic_content.append([])
        
    for i in range(len(MediacloudData)):
        topic_content[document_topics[i]].append(MediacloudData[i].__dict__)
    
    import json
    n=0
    for content in topic_content:
        f = open(os.path.dirname(__file__) + "/results/cluster"+str(n)+".json",'w')
        json.dump(sorted(content,  key=lambda sentence: sentence["similarity_to_topic"], reverse = True),f)
        f.close()
        n=n+1