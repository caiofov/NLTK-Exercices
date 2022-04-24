import nltk
from nltk.corpus import gutenberg as gt, webtext as wt, nps_chat, brown

emma = gt.words("austen-emma.txt") #returns all words from the text

#print(len(emma)) #number of words

emma_text = nltk.Text(emma) #we need to convert to a text object of the library to use it with some functions
#print(emma_text.concordance("surprize"))

def show_file_information(fileid): #show some informations about a given text
  num_chars = len(gt.raw(fileid)) #raw -> all characters of the text. Including spaces and special characters.
  num_words = len(gt.words(fileid))
  num_sents = len(gt.sents(fileid)) #sents -> divide the text in sentences
  num_vocab = len(set(w.lower() for w in gt.words(fileid)))

  avg_word_length = round(num_chars/num_words) #average word length
  avg_sentence_length = round(num_words/num_sents) #average sentence length
  avg_word_freq = round(num_words/num_vocab) #average word frequency
  
  print(avg_word_length, avg_sentence_length, avg_word_freq, fileid)

#show information of each file
'''
for fileid in gt.fileids():
  show_file_information(fileid)
'''

#WEBTEXT

def show_first_characters(fileid, num = 65): #show the first characters of a web text (65 by default)
  print(fileid, wt.raw(fileid)[:num], '...') #raw -> "real" text

'''
for fileid in wt.fileids():
  show_first_characters(fileid)
'''

#NPS CHAT - corpus of instant messaging chat sessions
chatroom = nps_chat.posts('10-19-20s_706posts.xml') #list of 'lists' (list of messages)

#BROWN CORPUS - first eletronic corpus
categories = brown.categories() #categories of the corpus

