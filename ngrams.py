#nltk
#scikit learn
from __future__ import print_function
from __future__ import division

from nltk import word_tokenize as tokenize
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.util import ngrams

from collections import Counter
import re


files = []
record = {}
for i in range(1,31):
  fname = "interviews/" + str(i) + ".txt"
  with open(fname, "r") as f:
    ff = []
    for line in f:
      if line != "\n" and "Introduction" not in line:
        l = line.decode('utf-8').split("\n")[0]
        l = re.sub(r'\([0-9]*\)','',l)
        l = re.sub(r'0xe2','',l)
        ff.append(l)
    files.append(ff)

for i,doc in enumerate(files):
  name = doc[0]
  record[i+1] = name

data = [" ".join(doc) for doc in files]
combined = " ".join(data)
c = re.sub('Oxe2','',combined)
tokens = tokenize(c)

#remove stopwords
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
print(len(tokens))
tokens = [token for token in tokens if token not in stop]
print(len(tokens))

bigrams = ngrams(tokens,2)
trigrams = ngrams(tokens, 3)

x = Counter(bigrams)
print(x.most_common(10))

#fdist = nltk.FreqDist(bigrams)
#for k,v in fdist.items():
#  print(k,v)

#get context of every word
#map to language
