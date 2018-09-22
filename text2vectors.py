
import numpy as np
import re

sample_text = """Natural language processing (NLP) is the ability of a computer
program to understand human language as it is spoken. NLP is a component of artificial intelligence (AI)."""

def wordTokens(sample_text):
    """Tokenize the sentence and returns a list of words"""
    
    words = re.findall(r'[A-Za-z]\w*',sample_text)
    words = set(words)
    return words

# Tokens contains the list of words
tokens = wordTokens(sample_text)
word_tokens = dict((word,i) for i,word in enumerate(tokens))

def oneHotEncoder(word_tokens):
  """Returns vectors for each token"""
  
    word_vectors = {}
    for i,word in enumerate(word_tokens):
        temp_vector = np.zeros((len(word_tokens)))
        temp_vector[i] = 1
        word_vectors[word] = temp_vector
    return word_vectors

wordvec = oneHotEncoder(word_tokens)

new_word = 'understand human language'
new_word_token = wordTokens(new_word)

# Converting the above sentence into a vector.
sen_vec = np.zeros((len(word_tokens)))
for word in new_word_token:
    new_word_vec = wordvec[word]
    sen_vec = sen_vec + new_word_vec

print("Sentence vector:",sen_vec)
