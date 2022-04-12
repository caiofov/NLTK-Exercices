import nltk
from nltk.corpus import gutenberg as gt, webtext as wt, nps_chat

emma = gt.words("austen-emma.txt") #returns all words from the text

#print(len(emma)) #number of words

emma_text = nltk.Text(emma) #we need to convert to a text object of the library to use it with some functions
#print(emma_text.concordance("surprize"))

def show_file_information(fileid): #show some informations about a given text
  num_chars = len(gt.raw(fileid)) #raw -> all characters of the text. Including spaces and special characters.
  num_words = len(gt.words(fileid))
  num_sents = len(gt.sents(fileid)) #sents -> divide the text in sentences
  num_vocab = len(set(w.lower() for w in gt.words(fileid)))

  avg_word_length = round(num_chars/num_words)
  avg_sentence_length = round(num_words/num_sents)
  avg_word_freq = round(num_words/num_vocab)
  print(avg_word_length, avg_sentence_length, avg_word_freq, fileid)

'''
for fileid in gt.fileids():
  show_file_information(fileid)
'''

#WEBTEXT

def show_first_characters(fileid, num = 65):
  print(fileid, wt.raw(fileid)[:num], '...')

for fileid in wt.fileids():
  show_first_characters(fileid)


#NPS CHAT
chatroom = nps_chat.posts('10-19-20s_706posts.xml')