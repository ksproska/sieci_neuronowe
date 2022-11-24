from keras.datasets.imdb import get_word_index
import tensorflow as tf


def load_dataset():
    return tf.keras.datasets.imdb.load_data(
        path="imdb.npz",
        num_words=None,
        skip_top=0,
        maxlen=None,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3
    )


def get_sentence(sentence_list):
    start_char = 1
    oov_char = 2
    index_from = 3
    word_index = get_word_index()
    inverted_word_index = dict(
        (i + index_from, word) for (word, i) in word_index.items()
    )
    inverted_word_index[start_char] = "[START]"
    inverted_word_index[oov_char] = "[OOV]"
    decoded_sequence = " ".join(inverted_word_index[i] for i in sentence_list)
    return decoded_sequence
