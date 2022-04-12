from nltk.book import *

def lexical_diversity(text): #returns the lexical diversity of a text
    return len((set(text))) / len(text)

def percentage(count, total):
    return 100*count/total

#CHAPTER 3

fdist1 = FreqDist(text1) #frequency distribution of each word
common = fdist1.most_common(50) #most common words in the text

#fdist1.plot(50,cumulative=True) #plots the most common words
#fdist1.hapaxes() #shows the hapaxes (words that happens only once)

def long_words(text, length = 15):
    return sorted([w for w in set(text) if len(w) > length])

def long_frequent_words(text, length = 7, ocurrency = 7):
    frequency_distribution = FreqDist(text)
    return sorted(w for w in set(text) if len(w) > length and frequency_distribution[w] > ocurrency)

#print(long_frequent_words(text5))

listed_bigrams = list(bigrams(['more', 'is', 'said', 'than', 'done'])) #pair of words

#print(listed_bigrams)

#collocations - frequent bigrams
#print(text4.collocations())

def frequent_lengths(text): #returns the frequency of each word length
  length_list = [len(w) for w in text]
  return FreqDist(length_list)

#frequent_lengths(text1).most_common() #most common lengths


#SOME PYTHON FUNCTIONS
'''
> s.startswith(t)	test if s starts with t
> s.endswith(t)	test if s ends with t
> t in s	test if t is a substring of s
> s.islower()	test if s contains cased characters and all are lowercase
> s.isupper()	test if s contains cased characters and all are uppercase
> s.isalpha()	test if s is non-empty and all characters in s are alphabetic
> s.isalnum()	test if s is non-empty and all characters in s are alphanumeric
> s.isdigit()	test if s is non-empty and all characters in s are digits
> s.istitle()	test if s contains cased characters and is titlecased (i.e. all words in s have initial capitals)
'''

