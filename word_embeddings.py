import utils
import random
import numpy as np
from collections import Counter


def get_text(file_name):
    """
        Reads text from the file and returns it.
    """

    with open(file_name, 'r') as file_read:
        text = file_read.read()

    return text


def get_words(text):
    """
        Reads the given text and returns less frequency words.
    """

    return utils.preprocess(text)

def word_mapping(words):

    vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
    int_words = [vocab_to_int[word] for word in words]

    return vocab_to_int, int_to_vocab, int_words


def subsampling(int_words):
    
    threshold = 1e-5
    word_counts = Counter(int_words)

    total_count = len(int_words)
    freqs = {word: count/total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}

    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

    return train_words



if __name__ == '__main__':

    FILE_NAME = 'sample.txt'

    text = get_text(FILE_NAME)

    print(text)





