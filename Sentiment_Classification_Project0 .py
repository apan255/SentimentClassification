import nltk
from nltk.corpus import movie_reviews
import random
import os
import sys
from nltk.corpus import stopwords
import re
import csv
import re
import math
import string
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from itertools import islice, izip
from collections import Counter

# define a feature definition function here
#########################################################################################
####  Pre-processing the documents  ####
#########################################################################################
def Pre_processing_documents(document):
	# "Pre_processing_documents"  
	# "create list of lower case words"
	word_list = re.split('\s+', document.lower())
	# punctuation and numbers to be removed
	punctuation = re.compile(r'[-.?!/\%@,":;()|0-9]')
	stop = stopwords.words('english')
	word_list = [punctuation.sub("", word) for word in word_list] 
	final_word_list = []
	for word in word_list:
		if word not in stop:
			final_word_list.append(word)
	stringword = " ".join(final_word_list)
	return stringword 




#########################################################################################
####  Features   accuracy calculation####
#########################################################################################
def Features_accuracy_calculation(featuresets):
	print "---------------------------------------------------"
	print "Training and testing a classifier "  
	training_size = int(0.1*len(featuresets))
	test_set = featuresets[:training_size]
	training_set = featuresets[training_size:]
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	print "Accuracy of classifier :"
	print nltk.classify.accuracy(classifier, test_set)
	print "---------------------------------------------------"
	print "Showing most informative features"
	print classifier.show_most_informative_features(50)
	print "---------------------------------------------------"
	print "Obtaining precision, recall and F-measure scores"
	Obtain_precision_recall_and_Fmeasure_scores(classifier,test_set)
	print ""

	
	
	
	
#########################################################################################
## Obtain precision, recall and F-measure scores. ##
#########################################################################################
def Obtain_precision_recall_and_Fmeasure_scores(classifier_type, test_set):
  reflist = []
  testlist = []
  for (features, label) in test_set:
	reflist.append(label)
	testlist.append(classifier_type.classify(features))
  print " "
  print "The confusion matrix"
  cm = nltk.metrics.ConfusionMatrix(reflist, testlist)
  print cm

#  precision and recall
# start with empty sets for true positive, true negative, false positive, false negative,

  (refpos, refneg, testpos, testneg) = (set(), set(), set(), set())
  
  for i, label in enumerate(reflist):
	if label == 'neg': refneg.add(i)
	if label == 'pos': refpos.add(i)
  for i, label in enumerate(testlist):
	if label == 'neg': testneg.add(i)
	if label == 'pos': testpos.add(i)
  
  def printmeasures(label, refset, testset):
	print label, 'precision:', nltk.metrics.precision(refset, testset)
	print label, 'recall:', nltk.metrics.recall(refset, testset)
	print label, 'F-measure:', nltk.metrics.f_measure(refset, testset)
  
  printmeasures('', refpos, testpos)
 

#########################################################################################
# returns two lists:  words in positive emotion class and
# words in negative emotion class
#########################################################################################
 

def read_words():
  poslist = []
  neglist = []
  # read all LIWC words from file
  wordlines = [line.strip() for line in open('liwcdic2007.dic')]
  # each line has a word or a stem followed by * and numbers of the word classes it is in
  # word class 126 is positive emotion and 127 is negative emotion
  for line in wordlines:
    if not line == '':
      items = line.split()
      word = items[0]
      classes = items[1:]
      for c in classes:
        if c == '126':
          poslist.append( word )
        if c == '127':
          neglist.append( word )
  return (poslist, neglist)
  
  
# test to see if a word is on the list
#   using a prefix test if the word is a stem with an *
# returns True or False
def isPresent(word, emotionlist):
  isFound = False
  # loop over all elements of list
  for emotionword in emotionlist:
    # test if a word or a stem
    if not emotionword[-1] == '*':
      # it's a word!
      # when a match is found, can quit the loop with True
      if word == emotionword:
        isFound = True
        break
    else:
      # it's a stem!
      # when a match is found, can quit the loop with True
      if word.startswith(emotionword[0:-1]):
        isFound = True
        break
  # end of loop
  return isFound

(poslist, neglist) = read_words()


##############################################
# "vary the representation of the subjectivity lexicon features ." 
##############################################

def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict
	
SLpath = "subjclueslen1-HLTEMNLP05.tff"
SL = readSubjectivity(SLpath) 	
############################################################
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
Baseline_performance_word_features = all_words.keys()[:180]
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document



##############################################
# representation of the negation features 
##############################################


negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']
negationwords.extend([ 'can', 'don', 't'])

stopwords_english = nltk.corpus.stopwords.words('english')
newstopwords = [word for word in stopwords_english if word not in negationwords]
new_all_words = [word for word in all_words if word not in newstopwords]
Negation_word_features = new_all_words[:2500]


print "##############################################"
print " Sentiment Classification Project Baseline feature 0           "
print "##############################################"

def Baseline_performance_document_features(document,c):
	document_words = set(document)
	features = {}
	for word in Baseline_performance_word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features
Baseline_performance_featuresets = [(Baseline_performance_document_features(d,c), c) for (d,c) in documents]
Features_accuracy_calculation(Baseline_performance_featuresets)




pre_processing_words = nltk.FreqDist(Pre_processing_documents(w) for w in movie_reviews.words())
processed_words_features = pre_processing_words.keys()[:1000]

print ""
print ""
print ""
print ""


print "##############################################"
print " Sentiment Classification Project combined feature  "
print "##############################################"

# define features that include word counts of subjectivity words
# negative feature will have number of weakly negative words +
#    2 * number of strongly negative words
# positive feature has similar definition
#    not counting neutral words
def SL_features(document, SL):
	document_words = set(document)
	# first add all the word features
	features = {}
	for word in processed_words_features:
		features['contains(%s)' % word] = (word in document_words)
		
	# then add features for count variables for the 4 classes of subjectivity
	weakPos = 0
	strongPos = 0
	weakNeg = 0
	strongNeg = 0
	
	
	for word in document_words:
		if word in SL:
			strength, posTag, isStemmed, polarity = SL[word]
			if strength == 'weaksubj' and polarity == 'positive':
				weakPos += 1
			if strength == 'strongsubj' and polarity == 'positive':
				strongPos += 1
			if strength == 'weaksubj' and polarity == 'negative':
				weakNeg += 1
			if strength == 'strongsubj' and polarity == 'negative':
				strongNeg += 1
			features['positcount'] = 2/3 * strongPos
			features['negativecount'] = 2/3 * strongNeg
			features['neutral'] = 1/2 * (weakPos  + weakNeg)
			
    #NOT Feature
	for word in Negation_word_features:
		features['contains(%s)' % word] = False
		features['contains(NOT%s)' % word] = False
        
    # go through document words in order
	for i in range(0, len(document)):
		word = document[i]
        if ((i + 1) < len(document)) and (word in negationwords):
            i += 1
            features['contains(NOT%s)' % document[i]] = (document[i] in processed_words_features)
		elif((i + 2) < len(document)) and (document[i+1] == "not"):
            i += 2
            features['contains(NOT%s)' % document[i]] = (document[i] in processed_words_features)
        else:
            if ((i + 3) < len(document)) and (word.endswith('n') and document[i+1] == "'" and document[i+2] == 't'):
                i += 3
                features['contains(NOT%s)' % document[i]] = (document[i] in processed_words_features)
            else:
                features['contains(%s)' % word] = (word in processed_words_features)
				
	return features

SL_featuresets = [(SL_features(d, SL), c) for (d,c) in documents]
Features_accuracy_calculation(SL_featuresets)
