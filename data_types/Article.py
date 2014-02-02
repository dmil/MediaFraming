from MediacloudData import MediacloudData
class Article(MediacloudData):
    
    def __init__(self, record):
        MediacloudData.__init__(self, record)
        self.text = record.get('download_text', None) 
        self.title = record.get('title', None) 
        self.description =  record.get('description', None) 


        
    def __str__(self):
        return ""
    
    def __repr__(self):
        return str(self.__dict__)
        
#class Article:
#   """Class to store an Article"""
#   def __init__(self, record):
#       self.download_text = record[0]
#       self.title = record[1]
#       self.url = record[2]
#       self.stories_id = record[3]
#       self.media_id = record[4]
#       self.guid = record[5]
#       self.description = record[6]
#       self.publish_date = str(record[7])
#       self.collect_date = str(record[8])
#       self.story_texts_id = record[9]
#       self.full_text_rss = record[10]
#       self.spidered = record[11]
#       self.medium_name = record[12]
#       self.medium_url = record[13]
