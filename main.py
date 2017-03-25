#main driver file

### STEP 1 - EXTRACT DATA
#################################################
from data_extractor import query_db, query_csv

#SELECT IF DATA IS IN CSV FILE
#raw_data = query_csv("/home/dhrumil/Desktop/MediaFraming/data/trayvon/trayvon_sentences.csv")

#SELECT IF DATA IS IN A POSTGRES DATABASE
query = "SELECT \
     	    trayvon_text.download_text, \
            trayvon_text.title, \
            trayvon_text.url, \
            trayvon_text.stories_id, \
            trayvon_text.media_id, \
            trayvon_text.guid, \
            trayvon_text.description,\
            trayvon_text.publish_date,\
            trayvon_text.collect_date, \
            trayvon_text.story_texts_id, \
            trayvon_text.full_text_rss, \
            trayvon_text.spidered, \
            trayvon_text.medium_name, \
            trayvon_text.medium_url \
            FROM  \
            public.trayvon_text;"
            
raw_data = query_db(host='localhost', dbname='trayvon2', username='dhrumil', password='xxxxxx')

### STEP 2 - CLEAN DATA
#################################################
from data_cleaner import make_sentences, make_articles

#SELECT IF DATA IS ARTICLES
articles = make_articles(raw_data)

#SELECT IF DATA IS SENTENCES
#sentences = make_sentences(raw_data)

#trayvon_sentences = []
#for sentence in sentences:
#    if 'trayvon' in sentence.text.lower():
#        trayvon_sentences.append(sentence)

### STEP 3 - CLUSTER DATA
#################################################
from cluster_maker import kmeans
from cluster_maker import latent_semantic_analysis
from cluster_maker import latent_drichlet_allocation
# type help(kmeans), help(latent_semantic_analysis), or help(latent_drichlet_allocation)
# into shell to learn possible different inputs for these functions & see full specifications

#kmeans(trayvon_sentences, num_clusters=2, stopwords='english', features='hashing')

#latent_drichlet_allocation(trayvon_sentences, 9)

### STEP 4 - ANALYZE CLUSTERS
#################################################
from cluster_analyzer import MediaSource, get_clusters, clutser_summary
#clusters = get_clusters('/home/dhrumil/Desktop/MediaFraming/results/')
#clutser_summary(clusters[3])
