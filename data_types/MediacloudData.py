from abc import ABCMeta, abstractmethod

class MediacloudData(object):
    """
    Abstract class - structures data coming out of mediacloud
    example implementations (Sentence, Paragraph, Article)
    """
    __metaclass__ = ABCMeta #defines class as abstract
    
    #classes that implement MediacloudData must have these methods
    #@abstractmethod
    
  

    def __init__(self, dictionary):
        def intget(tag):
            ret_val = dictionary.get(tag, None) 
            if type(ret_val)!=type(None): 
                ret_val = int(ret_val)
            return ret_val
            
        self.url = dictionary.get('url', None) 
        self.stories_id = intget('stories_id')
        self.media_id = intget('media_id')
        self.publish_date = dictionary.get('publish_date', None)
        self.guid = dictionary.get('guid', None)
        self.text = dictionary.get('text', "")#testing
        

    
    #@abstractmethod
    def __str__(self): pass
    
    #@abstractmethod
    def __repr__(self): pass

    #classes that implement MediacloudData must have these variables
    
    ############
    ## insert code here to require self.text (not sure how to do this yet)
    ## maybe that code goes in the "__init__" abstract method
    ############
    