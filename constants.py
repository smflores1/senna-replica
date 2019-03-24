import os
import itertools

# File names:
WDVEC_DETAILS = 'wdvec_details.json'
MODEL_DETAILS = 'model_details.json'

# Directories
DIR_WDVECS = 'wdvecs'
DIR_MODELS = 'models'
DIR_SAVED_MODEL = 'saved_model'

# File paths:
PATH_ROOT = os.getcwd()

# Lists:
LIST_PRED_METHOD_KEYS = ['true', 'dumb', 'rand']
LIST_TRAIN_TESTA_TESTB_KEYS = ['train', 'testa', 'testb']
LIST_TAG_TYPES = ['LOC', 'MISC', 'PER', 'ORG']
LIST_PERFORMANCE_KEYS = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
LIST_KEYWORD_TAGS = ['O-'] + [IOBES_tag + '-' + type_tag for IOBES_tag, type_tag in itertools.product(['I', 'B', 'E', 'S'], LIST_TAG_TYPES)]