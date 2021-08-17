import os
import json
import numpy as np
import keras
import tensorflow as tf
# from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
# from torchMoji.torchmoji.model_def import torchmoji_emojis


def get_data_file_path(relative_path):
    d = os.getcwd()
    # relative_path = r'app/data/' + relative_path
    file_path = os.path.join(d, relative_path)

    return file_path




model_path = './my_model.h5'

# Returns the indices of the k largest elements in array.


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


class Astolphise:
    def __init__(self):
        self.model = keras.models.load_model(model_path)

    def predict(self, image):
        stuff = self.model.predict(image)
        return stuff

