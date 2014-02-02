"""data_cleaner.py - takes records (lists of dicts) and returns specific types"""        

from data_types.Sentence import Sentence
from data_types.Paragraph import Paragraph
from data_types.Article import Article

def make_sentences(records):
    sentences = [] 
    for record in records:
        sentences.append(Sentence(record))
    return sentences
    
def make_paragraphs(records): 
    paragraphs = [] 
    for record in records:
        paragraphs.append(Paragraph(record))
    return paragraphs
    
def make_articles(records): 
    articles = [] 
    for record in records:
        articles.append(Article(record))
    return articles
