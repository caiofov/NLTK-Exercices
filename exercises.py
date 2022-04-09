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

print(long_frequent_words(text5))