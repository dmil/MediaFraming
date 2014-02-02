"""data_extractor.py - extracts sentences from the CSV file or Database, returns a dict"""

import psycopg2
from itertools import izip
import csv
import sys

#get data from DB or CSV into __dict__
def query_db(host, dbname, username, password):
    """returns a list of dictionaries (each one represents an entry in the DB)""" 
    
    #build connection string
    conn_string = "host='%s' dbname='%s' user='%s' password='%s'" % (host, dbname, username, password)
    
    #(debug) print the connection string we will use to connect
    print "Connecting to database\n	->%s" % (conn_string)
 
    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    # build query - EDIT THIS AS NEEDED TO QUERY YOUR DB
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
    
    def query_to_dicts(query_string, *query_args):
        """Run a simple query and produce a generator
        that returns the results as a bunch of dictionaries
        with keys for the column values selected.
        """
        cursor = conn.cursor() # conn.cursor will return a cursor object, you can use this cursor to perform queries
        cursor.execute(query_string, query_args)
        col_names = [desc[0] for desc in cursor.description]
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            row_dict = dict(izip(col_names, row))
            yield row_dict
        return
    
    #get list of records (dicts)
    record_generator = query_to_dicts(query)
    records = []
    for record in record_generator:
        records.append(record)

    return records
    
def query_csv(filepath):
    """returns a list of dictionaries (each one represents an entry in the CSV file)""" 
    csv.field_size_limit(sys.maxsize)
    
    records = []
    with open(filepath, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        field_names = csv_reader.next() 
        
        for row in csv_reader:
            record = {}
            for field_name,item in izip(field_names,row):
                record[field_name] = item             
            records.append(record)

    return records    