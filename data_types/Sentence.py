"""/mediaframing/data_types/Sentence"""
from MediacloudData import MediacloudData

class Sentence(MediacloudData): 
   """
   - stories_id - (int) id of the Article this Sentence came from 
   - url - (string)
   - media_id - (int) id of the media source this Sentence came from
   - publish_date - (string) date of publication
   - text - (string) text of the sentence
   - sentence_number - (int) number of this sentence in the Article
   """
   
       
   def dict_to_sentence(self,sentence_dict):
       """
       populate variables if input to __init__ was a dict
       if the dict did not contain any of these feilds, they are set to None
       """
       def intify(datum):
           if datum == None: return None
           else: return int(datum)
           
       self.text = sentence_dict.get('sentence', None) 
       self.sentence_number = intify(sentence_dict.get('sentence_number', None))
       
   def __init__(self, sentence=None):
       """constructor - sentence variable should be a dict"""
       MediacloudData.__init__(self, sentence)
       if type(sentence) == type({}):
           self.dict_to_sentence(sentence)
       else:
           raise TypeError('Error: Sentence constructor can only take a dict')
    
   def __repr__(self):
        """dict (JSON) representation of output for another python interpreter"""
        
        return str({
        "stories_id":self.stories_id,
        "url":self.url,
        "media_id": self.media_id,
        "publish_date": self.publish_date,
        "text": self.text,
        "sentence_number":self.sentence_number
        })
  
   def __str__(self): 
       """pretty output for printing"""
       
       print_string = "stories_id:" + str(self.stories_id) + "\n"
       print_string += "url:" + str(self.url) + "\n"
       print_string += "media_id:" + str(self.media_id) + "\n"
       print_string += "publish_date:" + str(self.publish_date) + "\n"
       print_string += "text:" + str(self.text) + "\n"
       print_string += "sentence_number:" + str(self.sentence_number) + "\n\n"
       return print_string
 
