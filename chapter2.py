import nltk
from nltk.corpus import gutenberg as gt, webtext as wt, nps_chat, brown, reuters, inaugural,udhr,stopwords

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

# plot_languages_udhr(['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik'])


#SECTION 2
#Counting Words by Genre
#freqDist -> simple input | ConditionalFreqDist -> pair of (condition, word)

def words_by_genre(genres):
    genre_word = [(gen,wrd) for gen in genres
                    for wrd in brown.words(categories = gen)
                    ]
    return genre_word
    

def count_words_by_genre(genres):
  genre_word = words_by_genre(genres)
  
  return len(genre_word)


#print(count_words_by_genre(['news', 'romance']))

cfd = nltk.ConditionalFreqDist(words_by_genre(["news","romance"]))
#cfd -> conditionalFreqDist with 2 conditions (news and romance)
#each of these conditions is just a frequency distribution
number_of_word_could_romance = cfd['romance']['could']


#2.3   Plotting and Tabulating Distributions

# The condition is either of the specified words, and the counts being plotted are the number of times the word occured in a particular speech. It exploits the fact that the filename for each speech, e.g., 1865-Lincoln.txt contains the year as the first four characters. This code generates the pair (word, year) for every instance of a word whose lowercased form starts with the word — such as "Americans" for "america" — in the file 1865-Lincoln.txt.

def words_in_inaugural(words):
  words_inaugural = [(target, fileid[:4]) 
      for fileid in inaugural.fileids()
      for w in inaugural.words(fileid)
      for target in words if w.lower().startswith(target)
      ]

  return words_inaugural

condition = nltk.ConditionalFreqDist(words_in_inaugural(['america','citizen']))
#condition.plot()

#length of the words in each language
def len_words_in_udhr(languages):
  len_words = [(lang, len(word))
                for lang in languages
                for word in udhr.words(lang + '-Latin1')
                ]
  return len_words

condition_udhr = nltk.ConditionalFreqDist(len_words_in_udhr(['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']))

#condition_udhr.plot()

#tabulate parameters -> when we omit, we get all of them
#condition_udhr.tabulate(conditions=['English','German_Deutsch'], samples=range(10), cumulative=True)


#2.4   Generating Random Text with Bigrams
#bigrams -> word pairs
#bigrams() -> takes a list of words and builds a list of consecutive word pairs. We need to use the list() function in order to see a list

sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
#print(list(nltk.bigrams(sent)))

#simple loop to generate a text
#word = initial content
def generate_model(cfdist, word, num=15):
  for i in range (num):
    print(word, end=' ')
    word = cfdist[word].max() #reset to be the most likely token in that context
  
text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text) #transform the text into bigrams
cfd = nltk.ConditionalFreqDist(bigrams)

#print(cfd['living']) #frequent words next to the word "living"

#generate_model(cfd, 'living')
#it is a simple approach that gets stuck in a loop

#4 - LEXICAL RESOURCES
#lexical entry -> consists of a headword (lemma) along with additional information such as the part of speech and the sense of definition
#homonyms -> two distinct words having the same spelling

#4.1 - Wordlist Corpora

#computes the vocabulary of a text, then removes all items that occur in an existing wordlist, leaving just the uncommon or mis-spelt words.
def unusual_words(text):
  text_vocab = set(w.lower() for w in text if w.isalpha())
  english_vocab = set(w.lower() for w in nltk.corpus.words.words())
  unusual = text_vocab - english_vocab
  return sorted(unusual)

#print(unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt')))

#stopwords - igh-frequency words like the, to and also that we sometimes want to filter out of a document before further processing

#print(stopwords.words('english'))

#fraction of words in a text are not in the stopwords list
def content_fraction(text):
  stopwords = stopwords.words('english')
  content = [w for w in text if w.lower() not in stopwords]
  return len(content) / len(text)


#PUZZLE - Points: 21 (good) | 32 (very good) | 42 (excellent)
# E G I
# V R V
# O N L
# Rules:
# 1. Words of four letter or more and must be at least one nine-letter
# 2. Each letter may be used once per word
# 3. Each word must contain the center letter (R)
# 4. No plurals ending in "s", no foreign words and proper names


def puzzle(letters, obligatory, min_length):
  puzzle_letters = nltk.FreqDist(letters) #frequency of each letter on the puzzle
  wordlist = nltk.corpus.words.words() #list of words

  return [w for w in wordlist if len(w) >= min_length #rule 1
                              and obligatory in w #rule 3
                              and nltk.FreqDist(w) <= puzzle_letters #rule 2
          ]

points = {'good': 21,'very good': 32, 'excellent': 42}

puzzle_words = puzzle('engivrvonl', 'r', 6)

#Name corpus
names = nltk.corpus.names
male_names = names.words('male.txt')
female_names = names.words('female.txt')
agender_names = [w for w in male_names if w in female_names]

ending_letters_per_gender = nltk.ConditionalFreqDist(
                                (fileid, name[-1])
                                for fileid in names.fileids()
                                for name in names.words(fileid))

#ending_letters_per_gender.plot()

#4.2 A Pronouncing Dictionary
#CMU Pronouncing Dictionary for US English
#which was designed for use by speech synthesizers.

entries = nltk.corpus.cmudict.entries() #all entries - list of tuples
#two parts: (word, pronunciation)

#checks if the last phones of the words match with the given one
def find_rhyming_words(syllable, entries = nltk.corpus.cmudict.entries()):
  number_of_phones = len(syllable)
  return [word for word, pron in entries 
                  if pron[(-1*number_of_phones):] == syllable]

#print(find_rhyming_words(['N', 'IH0', 'K', 'S']))

#The phones contain digits to represent 
#primary stress (1), secondary stress (2) and no stress (0)
def stress(pron):
  return [char for phone in pron for char in phone if char.isdigit()]

stress_pattern1 = [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
stress_pattern2 = [w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]


#we can also access the words by a dictionary (data structe)
prondict = nltk.corpus.cmudict.dict()
#print(prondict)

def text_to_speech(sentence, pronunciation_dictionary = nltk.corpus.cmudict.dict()):
  sentence_list = sentence.split(' ')
  return [ph for w in sentence_list 
                  for ph in prondict[w][0]]

#print(text_to_speech('natural language processing'))

#4.3 Comparative Wordlists
# Swadesh wordlists -> 200 common words in several languages

languages = nltk.corpus.swadesh.fileids()

words_english = nltk.corpus.swadesh.words('en')

#with "entries" we can access congnate words from multiple languages
french_to_english = nltk.corpus.swadesh.entries(['fr','en'])
#transforming the word list into dictionary, so we can access the items easily
translate = dict(french_to_english)
chien_english = translate['chien'] #example of use

#inserting more languages in our translator
german_to_english = nltk.corpus.swadesh.entries(['de','en'])
spanish_to_english = nltk.corpus.swadesh.entries(['es','en'])

translate.update(dict(german_to_english))
translate.update(dict(spanish_to_english))

#compare words in various languages
def compare_words(languages, word_numbers):
  return [nltk.corpus.swadesh.entries(languages)[i] for i in word_numbers]

#print(compare_words(['en', 'de', 'nl', 'es', 'fr', 'pt', 'la'], [139, 140, 141, 142]))

#4.4 Shoebox and Toolbox Lexicons
