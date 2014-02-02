"""cluster_analyzer.py contains algorithms that can analyze a 
set of results (directory with cluster json files)"""

"""clusteranalyzer.py analyzes the clusters output into json files"""

import os
from data_types.Sentence import Sentence
from data_types.MediacloudData import MediacloudData
#from sentences_datacleaner import Sentence, Paragraph
from dateutil import parser
import numpy as np
from matplotlib.dates import date2num, num2date
import string
from nltk.tokenize import wordpunct_tokenize
import operator
from nltk.corpus import stopwords
import json, csv

class MediaSource:
   'Class to represent media sources'
   count = 0
   
   def __init__(self, mid = None, mt = None, pv = None, cid = 1):
       self.controversies_id = cid
       self.media_id = mid
       self.media_type = mt
       self.political_valence = pv

       MediaSource.count += 1

class Cluster:
   '''Class to store a cluster'''
   count = 0
   
   def __init__(self, json_list, filepath=None):
       self.sentences = []
       self.length = len(json_list)
      
       self.filepath = filepath
       self.filename = os.path.basename(filepath)
       
       publish_dates_numeric = []
       for sentence_dict in json_list:
           this_sentence = MediacloudData(sentence_dict) 
           #Sentence variables: stories_id, url, media_id, publish_date, text, sentence_number
           self.sentences.append(this_sentence)
           
           #get the publish date
           publish_dates_numeric.append(date2num(parser.parse(this_sentence.publish_date)))
       
       self.average_publish_date = num2date(np.mean(publish_dates_numeric))
       Cluster.count += 1 
   
   def __str__(self):
       pretty_string = "\n"
       pretty_string += str(self.filename) + "\n"
       pretty_string += "filepath: " + str(self.filepath) + "\n"
       pretty_string += "Number of Items: " + str(self.length) + "\n"
       pretty_string += "Avg. Publish Date: " + str(self.average_publish_date) + "\n"
       #pretty_string += "Sentences: " + str(self.sentences) + "\n"
       return pretty_string


def load_mediasources():
    media_sources = {} #Key=media_id Value=mediasource
    with open(os.path.dirname(__file__) + "/data/trayvon/trayvon_controversy_media_codes.csv", 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for item in spamreader:
            if item[1] in media_sources.keys(): #if there is an entry
                if item[2] == 'media_type':
                    media_sources[item[1]].media_type = item[3]
                elif item[2] =='political_valence':
                    media_sources[item[1]].political_valence = item[3]
            else:
                media_sources[item[1]] = MediaSource(item[1])
                if item[2] == 'media_type':
                    media_sources[item[1]].media_type = item[3]
                elif item[2] =='political_valence':
                    media_sources[item[1]].political_valence = item[3]
    return media_sources

def frequencies(sentence_texts, stopword = False):
    #lower case
    out = sentence_texts.lower()
    
    #remove punctuation
    out = out.translate(string.maketrans("",""), string.punctuation) 
    
    #tokenize     
    out = wordpunct_tokenize(out) 
    
    #build Dictionary of key=word value=number of occurances
    frequencies = {}
    for word in out:
        if word not in frequencies:
            #if word is a stopword and stopword is on, do not add
            if not(stopword == True and word in stopwords.words('english')):
                frequencies[word] = 1
        else:
            frequencies[word] += 1
    
    #sort frequencies
    sorted_frequencies = sorted(frequencies.iteritems(), key=operator.itemgetter(1), reverse=True)

    #output largest frequency first
    return sorted_frequencies
    
def get_clusters(folder_path):
    #Get files in /results folder
    #folder_path = os.path.dirname(__file__)
    root_path, folder_names, file_names = next(os.walk(folder_path))
    #print root_path
    #print folder_names
    #print file_names
    #create list of cluster objects
    clusters = [] 
    for fname in sorted(file_names):
        #read from file
        f = open(root_path + fname,'r')
        #print root_path+fname
        cluster_dict=eval(f.read().replace("null", "None"))
        f.close()
    
        #create Cluster object
        cluster = Cluster(cluster_dict, filepath = root_path + fname)
        
        #append to list
        clusters.append(cluster)   
        
    return clusters

def clutser_summary(cluster):
    all_text = ""
    print cluster
    for sentence in cluster.sentences:
        all_text += sentence.text + " "
    word_frequencies = frequencies(all_text, True)
                
    #output to console
    #print cluster.filename
    #print "# of Sentences: " + str(cluster.length) 
    #print "Average publish date: " + str(cluster.average_publish_date)
    print word_frequencies[0:50]
               
    #do media_source analysis
    media_source_analyis(cluster.filepath)
        
    print ""
    #print len(word_frequencies)
        

def media_source_analyis(fp):
    #enter cluster path here
    file_path = open(fp,'r')
    
    j_data = json.load(file_path)
    ms = load_mediasources()
    
    num_of_items = len(j_data)    

    print
    no_valence = 0
    conservative = 0
    liberal = 0
    neutral = 0
    for item in j_data:
        if str(item['media_id']) in ms.keys():
            pv = ms[str(item['media_id'])].political_valence
            if pv == None:
                no_valence += 1
            elif pv == 'Conservative':
                conservative += 1
            elif pv == 'Liberal':
                liberal += 1
            elif pv == 'Neutral':
                neutral += 1
            else:
                no_valence += 1
        else:
            no_valence += 1
    print "Percent Items with Identifiable Political Valance: " +"{0:.2f}%".format(float((num_of_items - no_valence))/ num_of_items * 100)
    with_valence = conservative + liberal + neutral
    
    if with_valence != 0:
        print "\t of those... Percent Conservative: " + "{0:.2f}%".format(float(conservative)/with_valence*100)
        print "\t of those... Percent Liberal: " + "{0:.2f}%".format(float(liberal)/with_valence*100)
        print "\t of those... Percent Neutral: " + "{0:.2f}%".format(float(neutral)/with_valence*100)

    print 
    media_types = {}
    for item in j_data:
        if str(item['media_id']) in ms.keys():
            mt = ms[str(item['media_id'])].media_type
            if mt in media_types.keys():
                media_types[mt] += 1
            else:
                media_types[mt] = 1
    #import pprint
    #pprint.pprint(media_types)
    total = 0
    for key in media_types.keys():
        total += media_types[key]
    print "Percent Items With ID-able Media Type: " + "{0:.2f}%".format(float(total)/num_of_items*100)
    for key in media_types.keys():
        print key[0:40] + " : " + "{0:.2f}%".format(float(media_types[key])/total*100)
    
      