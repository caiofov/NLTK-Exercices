import nltk
from nltk.corpus import gutenberg as gt, webtext as wt, nps_chat, brown, reuters, inaugural,udhr

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

words_news = brown.words(categories = "news") #words from the news categories
file_cg22 = brown.words(fileids=["cg22"]) #words from the file specified
some_sentences = brown.sents(categories=['news', 'editorial', 'reviews']) #list of sentences of the categories specified


#stylistics = systematic differences between genre
def frequency_distribution_category(category_name, modals): #prints the frequency of each given word in a given genre
  words_category = brown.words(categories=category_name)
  fdist = nltk.FreqDist(w.lower() for w in words_category)
  
  for m in modals:
    print(m + ':', fdist[m], end=' ')

#frequency_distribution_category("news", ['can', 'could', 'may', 'might', 'must', 'will'])

def frequency_distribution_categories(categories_names, modals): #the same as the previous one, but for more genres
  cfd = nltk.ConditionalFreqDist(
            (genre, word)
            for genre in brown.categories()
            for word in brown.words(categories=genre))
  cfd.tabulate(conditions=categories_names, samples=modals) #prints the given data in table form

#frequency_distribution_categories(['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor'], ['can', 'could', 'may', 'might', 'must', 'will'])


#REUTERS CORPUS - The documents have been classified into 90 topics, and grouped into two sets, called "training" and "test"
reuters_fileids = reuters.fileids()
reuters_categories = reuters.categories()

training9865_categories = reuters.categories('training/9865') #a single document can have more than 1 category
#it alsos accepts a list of documents as parameters

files_barley_category = reuters.fileids('barley') #returns all file ids in barley category
#it can be given a list of categories too as well

list_words = reuters.words('training/9865') #also accepts a list of file ids


#INAUGURAL ADDRESS CORPUS
inaugural_fileids = inaugural.fileids()
#the first 4 characters of each file id is the year of the text
def get_year_inaugural_file(fileids):
  print([fileid[:4] for fileid in inaugural.fileids()])

#get_year_inaugural_file(inaugural_fileids)

def plot_year_distribution_inaugural_file(targets):
  cfd = nltk.ConditionalFreqDist(
           (target, fileid[:4])
           for fileid in inaugural.fileids()
           for w in inaugural.words(fileid)
           for target in targets
           if w.lower().startswith(target))
  cfd.plot()

#plot_year_distribution_inaugural_file(['america', 'citizen'])


#UNIVERSAL DECLARATION OF HUMAN RIGHTS

udhr_fileids = nltk.corpus.udhr.fileids()

def plot_languages_udhr(languages):
  cfd = nltk.ConditionalFreqDist(
           (lang, len(word))
           for lang in languages
           for word in udhr.words(lang + '-Latin1'))
  cfd.plot(cumulative=True)

plot_languages_udhr(['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik'])