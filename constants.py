import os
import string
import itertools
import numpy as np

# Text file names:
TXT_WORDS_LIST = 'words_list.txt'
TXT_EMBEDDINGS = 'embeddings.txt'

# Json file names:
JSON_MODEL_DETAILS = 'model_details.json'
JSON_TRAIN_DETAILS = 'train_details.json'

# Directories
DIR_CORPUS = 'corpus'
DIR_MODELS = 'models'
DIR_WDVECS = 'wdvecs'
DIR_SAVED_MODEL = 'saved_model'

# File paths:
PATH_ROOT = os.getcwd()

# Lists:
LIST_TAG_TYPES = ['LOC', 'MISC', 'PER', 'ORG']
LIST_PRED_METHOD_KEYS = ['true', 'dumb', 'rand']
LIST_TRAIN_TESTA_TESTB_KEYS = ['train', 'testa', 'testb']
LIST_TRAIN_TESTA_TESTB_TXTS = ['train.txt', 'testa.txt', 'testb.txt']
LIST_PERFORMANCE_KEYS = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
LSIT_PUNCTUATION = list(string.punctuation) + ['``', "''", '--', '..', '...', '....', '—']
LIST_KEYWORD_TAGS = ['O-'] + [IOBES_tag + '-' + type_tag for IOBES_tag, type_tag in itertools.product(['I', 'B', 'E', 'S'], LIST_TAG_TYPES)]
LIST_ONE_HOT_ENCODE = {LIST_KEYWORD_TAGS[i]: np.identity(len(LIST_KEYWORD_TAGS))[i,:] for i in range(len(LIST_KEYWORD_TAGS))}

# Miscellaneous:
CAPS_DIM = 4
