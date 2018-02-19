#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:46:54 2017

@author: macintosh
"""

import re

import math
import io
from textblob import TextBlob as tb




from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from collections import defaultdict


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def convert(fname, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)
    
    output = io.StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(fname, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close
    return text



doc1 = convert("doc1.pdf")
doc2 = convert("doc2.pdf")
doc3 = convert("doc3.pdf")
doc4 = convert("doc4.pdf")
doc5 = convert("doc5.pdf")
doc6 = convert("doc6.pdf")
doc7 = convert("doc7.pdf")
doc8 = convert("doc8.pdf")



import nltk
allDoc = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8]

cleanDoc = []

for i in allDoc:    
    lowers = i.lower()


#remove punctuation
    s = re.sub(r'[^\w\s]','',lowers)
    k = re.sub(r"(^|\W)\d+($|\W)", ' ', s)
    c = re.sub('\s+',' ',k).strip()



    tokens = nltk.word_tokenize(c)



    filtered = [w for w in tokens if not w in stopwords.words('english')]



# =============================================================================
#     ps = PorterStemmer()
# 
# 
#     stemmedList = []
#     for w in filtered:
#         stemmedList.append((ps.stem(w)))
#  
#     asd = [' '.join(stemmedList)]
# =============================================================================
    
    cleanDoc.append(' '.join(filtered))

cleaned = [''.join(h) for h in cleanDoc]

from __future__ import division
import string
import math
import collections

tokenize = lambda doc: doc.lower().split(" ")

all_documents = cleaned

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values


tfdocs = []
def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append([term,tf * idf[term]])
            tfdocs.append([term,tf])
    return doc_tfidf

tdidfdocs = tfidf(all_documents)


tdidfdocs.sort(key = lambda x : x[1], reverse = True)

tfdocs.sort(key = lambda x : x[1], reverse = True)





file = open('tfidfdocs.csv', 'w', encoding = 'utf-8')
for t in tdidfdocs[:50]:

    file.write("%s\t%s\n" % (str(t[0]), str(t[1])))
        
        
file.close()


file2 = open('tfdocs.csv', 'w', encoding = 'utf-8')
for t in tfdocs[:50]:

    file2.write("%s\t%s\n" % (str(t[0]), str(t[1])))
        
        
file2.close()




# =============================================================================
# 
# 
# 
# default_data={}
# for value in tdidfdocs:
#     default_data[value[0]] = value[1]
#     
# 
# 
# 
# top_50 = lambda :""
# top_50.values = tdidfdocs
# d={}
# for a, x in top_50.values:
#    d[a] = 8-x
# 
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# 
# wordcloud = WordCloud()
# wordcloud.generate_from_frequencies(frequencies=d)
# plt.figure( figsize=(5,3) )
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
# 
# 
# =============================================================================
from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud, STOPWORDS

text = ' '.join(i[0] for i in tdidfdocs[:50])
wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                      relative_scaling = 1.0,
                      stopwords = {'to', 'of'} # set or space-separated string
                      ).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloudtfidf")
#plt.show()g


from os import path
from scipy.misc import imread
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud, STOPWORDS

text = ' '.join(i[0] for i in tfdocs[:50])
wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                      relative_scaling = 1.0,
                      stopwords = {'to', 'of'} # set or space-separated string
                      ).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloudtf")
#plt.show()




"""
#in Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
print(sklearn_representation)
"""




"""
########### END BLOG POST 1 #############

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

tfidf_representation = tfidf(all_documents)
our_tfidf_comparisons = []
for count_0, doc_0 in enumerate(tfidf_representation):
    for count_1, doc_1 in enumerate(tfidf_representation):
        our_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

skl_tfidf_comparisons = []
for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
    for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
        skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

for x in zip(sorted(our_tfidf_comparisons, reverse = True), sorted(skl_tfidf_comparisons, reverse = True)):
    print (x)
"""

"""
import nltk

from nltk import word_tokenize
from nltk.collocations import BigramCollocationFinder


import collections
from nltk.tokenize import word_tokenize
a = []
b = []
bigram_measures = nltk.collocations.BigramAssocMeasures()


with open('grafiker.txt', 'r', encoding = 'utf-8') as myfile:
    data=myfile.read()


    splittedAsd = data.split("%asd%")
    bigrams = [x for l in splittedAsd for x in zip(l.split(" ")[:-1], l.split(" ")[1:])]   
    for i in bigrams:
        k = low(str(i))
        b.append(k)
# =============================================================================
#     
#     for j in splittedAsd:
# 
# 
#     
#         finder = BigramCollocationFinder.from_words(word_tokenize(i))
#         a.append(finder.nbest(bigram_measures.pmi, 100))
#         
#         print(finder.nbest(bigram_measures.pmi, 100))      
#         frequencies = collections.Counter(splittedAsd)
#         print(frequencies.most_common(10))
#         
#         
# =============================================================================

        
        

# =============================================================================
# vergiBigram = []        
#          
# for i in b:
#     for j in i:
#         for k in j:
#             vergiBigram.append(list(j))
#                  
#  
#  
# vergiB = []
# for jk in vergiBigram:
#  
#     aas = " ".join(jk)
#     vergiB.append(aas)
# 
# =============================================================================

for h in b:
    print(b.count(h))
    print(h)
    
    
    

count2 = 0        
for i in splittedAsd:
    if "torna freze" in i:
        count2 = count2 + 1
        


from nltk.tokenize import sent_tokenize, word_tokenize


 
# =============================================================================
# vocabulary = vergiB
# word_index = {w: idx for idx, w in enumerate(vocabulary)}
#  
# VOCABULARY_SIZE = len(vocabulary)
# DOCUMENTS_COUNT = len(splittedAsd)
#  
# print(VOCABULARY_SIZE, DOCUMENTS_COUNT)
# 
# =============================================================================



sortedIdf = []

from collections import defaultdict
import math


word_idf = defaultdict(lambda: 0)

 
for word in b:
    word_idf[word] = math.log(841 / float(1 + b.count(word)))
 
#print(word_idf['bölüm mezun'])     # 7.49443021503
#print(word_idf['ölçüm alet'])     # 3.61286641709
for i in word_idf:
    
    sortedIdf.append([i,word_idf[i]])
print(i +"\t       "+ str((word_idf[i])))
    
sortedIdf.sort(key = lambda ele : ele[1])
"""