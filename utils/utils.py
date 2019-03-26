import os
import time
import numpy as np
import tensorflow as tf
import constants as const

class WdVecModel(object):

    def __init__(self):

        self.wv = {}
        self.vector_size = 0

class Word2Vec(object):

    @classmethod
    def load(cls, path):

        wdvec_model = WdVecModel()

        words_list_path = os.path.join(path, const.TXT_WORDS_LIST)
        embeddings_path = os.path.join(path, const.TXT_EMBEDDINGS)

        with open(words_list_path, 'r') as token_handle, open(embeddings_path, 'r') as vector_handle:

            for token, vector in zip(token_handle, vector_handle):
                float_vector = [float(x) for x in vector.rstrip('\n').split(' ')]
                wdvec_model.wv[token.rstrip('\n')] = np.array(float_vector)

        vocab_list = list(wdvec_model.wv.keys())

        wdvec_model.vector_size = len(wdvec_model.wv[vocab_list[0]])
        assert all([len(wdvec_model.wv[word]) == wdvec_model.vector_size for word in vocab_list])

        return wdvec_model

def default(num):

    '''
    
    Description:

    Inputs:

    Output:

    '''

    if isinstance(num, np.int64):
        return int(num)
    raise TypeError

def check_if_token_is_number(token):

    mod_token = token.lower()
    for char in const.LSIT_PUNCTUATION + ['st', 'nd', 'rd', 'th', 's']:
        mod_token = mod_token.replace(char, '')
    if mod_token.isdigit():
        token = 'number'

    return token

def get_text_embedding(padding,
                       wdvec_model,
                       sentences_list):

    inputs_dict = {}

    for sentnc_counter, tok_list in enumerate(sentences_list):

        vec_list = []
        tok_list = ['PADDING'] * padding + tok_list + ['PADDING'] * padding
        tok_list = [check_if_token_is_number(token) for token in tok_list]

        for token in tok_list:

            caps_vec = caps_vec_map(token)

            if token.lower() in wdvec_model.wv:
                word_vec = wdvec_model.wv[token.lower()]
            else:
                word_vec = wdvec_model.wv['UNKNOWN']

            vec_list += [np.concatenate([word_vec, caps_vec])] 

        # Form the context vector:
        inputs_vec_list = []
        for i in range(padding, len(vec_list) - padding):
            inputs_vec_list += [np.concatenate(vec_list[i - padding: i + padding + 1])]

        inputs_dict['sentence_' + str(sentnc_counter)] = np.stack(inputs_vec_list)
        tok_list = []

    # Shape the data:
    data_dict = {}
    for key in inputs_dict: # 'key' indexes the sentences.
        data_dict[key] = {'inputs': inputs_dict[key]}

    return data_dict

def get_corpus_embedding(padding,
                         corpus_path,
                         wdvec_model,
                         tag_type = 'IOB'):

    start = time.time()

    entity_counter = 0
    nonent_counter = 0

    inputs_dict = {}
    target_dict = {}
    for data_dict in [inputs_dict, target_dict]:
        for key in const.LIST_TRAIN_TESTA_TESTB_KEYS:
            data_dict[key] = {}

    tag_count_dict = {}

    data_dict = {}
    data_details_dict = {'padding': padding}
    for key_1, file in zip(const.LIST_TRAIN_TESTA_TESTB_KEYS,
                           const.LIST_TRAIN_TESTA_TESTB_TXTS):

        file_path = os.path.join(corpus_path, file)
        with open(file_path) as file_handle:

            inputs_dict[key_1] = {}
            target_dict[key_1] = {}

            tag_count_dict[key_1] = {tag: 0 for tag in const.LIST_KEYWORD_TAGS}

            tok_list = []
            tag_list = []

            entity_counter = 0
            nonent_counter = 0
            sentnc_counter = 0

            for line in file_handle:

                line = line.split(' ')

                if len(line) == 2:

                    token = line[0]

                    token = check_if_token_is_number(token)

                    ne_tag = line[1].rstrip('\n')
                    if ne_tag == 'O': ne_tag = 'O-'
                    ne_tag = ne_tag.split('-')

                    tok_list += [token]
                    tag_list += [ne_tag]

                elif line == ['\n'] and tok_list != []:

                    key_2 = 'sentence_' + str(sentnc_counter)
                    sentnc_counter += 1

                    # Change tagging scheme from IOB to IOBES if it is not already the latter:
                    if tag_type == 'IOB':
                        IOBES_tag_list = []
                        IOB_tag_list = [['O', '']] + tag_list + [['O', '']]
                        for i in range(1, len(IOB_tag_list) - 1):
                            tag_window = IOB_tag_list[i - 1: i + 2]
                            IOBES_tag_list += [map_to_IOBES_tag(tag_window)]
                        tag_list = list(IOBES_tag_list)
                    else:
                        assert tag_type == 'IOBES'

                    vec_list = []
                    tok_list = ['PADDING'] * padding + tok_list + ['PADDING'] * padding

                    for token in tok_list:

                        caps_vec = caps_vec_map(token)

                        if token.lower() in wdvec_model.wv:
                            word_vec = wdvec_model.wv[token.lower()]
                        else:
                            word_vec = wdvec_model.wv['UNKNOWN']

                        vec_list += [np.concatenate([word_vec, caps_vec])] 

                    # Rejoin the tag position and type with a dash and add the sentence padding:
                    tag_list = [IOBES_tag + '-' + type_tag for [IOBES_tag, type_tag] in tag_list]
                    tag_list = ['O-'] * padding + tag_list + ['O-'] * padding

                    # Form the context vector:
                    inputs_vec_list = []
                    target_vec_list = []
                    for i in range(padding, len(vec_list) - padding):

                        inputs_vec_list += [np.concatenate(vec_list[i - padding: i + padding + 1])]
                        target_vec_list += [const.LIST_ONE_HOT_ENCODE[tag_list[i]]]

                        if tag_list[i] == 'O-':
                            nonent_counter += 1
                        else: 
                            entity_counter += 1

                        tag_count_dict[key_1][tag_list[i]] += 1

                    inputs_dict[key_1][key_2] = np.stack(inputs_vec_list)
                    target_dict[key_1][key_2] = np.stack(target_vec_list)

                    word_counter = entity_counter + nonent_counter

                    tok_list = []
                    tag_list = []

        # Shape the data:
        data_dict[key_1] = {}
        for key_2 in inputs_dict[key_1]: # key_2 indexes the sentences.
            data_dict[key_1][key_2] = {'inputs': inputs_dict[key_1][key_2],
                                       'target': target_dict[key_1][key_2]} 

        # Record the data curation details:
        data_details_dict['num_' + key_1 + '_datapoints'] \
        = sum([data_dict[key_1][key_2]['inputs'].shape[0] for key_2 in data_dict[key_1]])
        data_details_dict['num_' + key_1 + '_sentences'] = len(data_dict[key_1])

        for tag in const.LIST_KEYWORD_TAGS:
            data_details_dict['num_' + key_1 + '_' + tag + '_tags'] = tag_count_dict[key_1][tag]

    train_data = np.concatenate([data_dict['train'][key_2]['inputs'] for key_2 in data_dict['train']])
    data_details_dict['translation'] = list(np.mean(train_data, axis = 0))
    data_details_dict['dilation'] = np.amax(np.std(train_data - data_details_dict['translation'], axis = 1))

    print('Finished mapping all tokens in the input data to vectors.')
    print('Total time for data curation:', round(time.time() - start, 3))

    return data_dict, data_details_dict

def map_to_IOBES_tag(tags_window):

    assert len(tags_window) == 3

    IOB_tag_list  = [tag[0] for tag in tags_window]
    type_tag_list = [tag[1] for tag in tags_window]

    assert len(IOB_tag_list) == 3
    assert len(type_tag_list) == 3
    assert all([IOB_tag in ['I', 'O', 'B'] for IOB_tag in IOB_tag_list])
    assert all([type_tag in const.LIST_TAG_TYPES + [''] for type_tag in type_tag_list])

    if IOB_tag_list == ['O', 'O', 'O']:
        return ['O', '']
    if IOB_tag_list == ['O', 'O', 'I']:
        return ['O', '']
    if IOB_tag_list == ['O', 'O', 'B']: # Should never happen.
        # print('Incorrect tagging detected:', tags_window)
        # return False
        return ['O', ''] 

    if IOB_tag_list == ['O', 'I', 'O']:
        return ['S', type_tag_list[1]]
    if IOB_tag_list == ['O', 'I', 'I']:
        if type_tag_list[1] == type_tag_list[2]:
            return ['B', type_tag_list[1]]
        else:
            return ['S', type_tag_list[1]]
    if IOB_tag_list == ['O', 'I', 'B']:
        return ['S', type_tag_list[1]]

    if IOB_tag_list == ['O', 'B', 'O']: # Should never happen.
        # print('Incorrect tagging detected:', tags_window)
        # return False
        return ['S', type_tag_list[1]]
    if IOB_tag_list == ['O', 'B', 'I']: # Should never happen.
        # print('Incorrect tagging detected:', tags_window)
        # return False
        return ['B', type_tag_list[1]]
    if IOB_tag_list == ['O', 'B', 'B']: # Should never happen.
        # print('Incorrect tagging detected:', tags_window)
        # return False
        return ['S', type_tag_list[1]]

    if IOB_tag_list == ['I', 'O', 'O']:
        return ['O', '']
    if IOB_tag_list == ['I', 'O', 'I']:
        return ['O', '']
    if IOB_tag_list == ['I', 'O', 'B']: # Should never happen.
        # print('Incorrect tagging detected:', tags_window)
        # return False
        return ['O', '']

    if IOB_tag_list == ['I', 'I', 'O']:
        if type_tag_list[0] == type_tag_list[1]:
            return ['E', type_tag_list[1]]
        else:
            return ['S', type_tag_list[1]]
    if IOB_tag_list == ['I', 'I', 'I']:
        if type_tag_list[0] == type_tag_list[1]:
            if type_tag_list[1] == type_tag_list[2]:
                return ['I', type_tag_list[1]]
            else:
                return ['E', type_tag_list[1]]
        else:
            if type_tag_list[1] == type_tag_list[2]:
                return ['B', type_tag_list[1]]
            else:
                return ['S', type_tag_list[1]]
    if IOB_tag_list == ['I', 'I', 'B']:
        if type_tag_list[0] == type_tag_list[1]:
            return ['E', type_tag_list[1]]
        else:
            return ['S', type_tag_list[1]]

    if IOB_tag_list == ['I', 'B', 'O']:
        return ['S', type_tag_list[1]]
    if IOB_tag_list == ['I', 'B', 'I']:
        if type_tag_list[1] == type_tag_list[2]:
            return ['B', type_tag_list[1]]
        else:
            return ['S', type_tag_list[1]]
    if IOB_tag_list == ['I', 'B', 'B']:
        return ['S', type_tag_list[1]]

    if IOB_tag_list == ['B', 'O', 'O']:
        return ['O', '']
    if IOB_tag_list == ['B', 'O', 'I']:
        return ['O', '']
    if IOB_tag_list == ['B', 'O', 'B']: # Should never happen.
        # print('Incorrect tagging detected:', tags_window)
        # return False
        return ['O', '']

    if IOB_tag_list == ['B', 'I', 'O']:
        if type_tag_list[0] == type_tag_list[1]:
            return ['E', type_tag_list[1]]
        else:
            return ['S', type_tag_list[1]]
    if IOB_tag_list == ['B', 'I', 'I']:
        if type_tag_list[0] == type_tag_list[1]:
            if type_tag_list[1] == type_tag_list[2]:
                return ['I', type_tag_list[1]]
            else:
                return ['E', type_tag_list[1]]
        else:
            if type_tag_list[1] == type_tag_list[2]:
                return ['B', type_tag_list[1]]
            else:
                return ['S', type_tag_list[1]]
    if IOB_tag_list == ['B', 'I', 'B']:
        if type_tag_list[0] == type_tag_list[1]:
            return ['E', type_tag_list[1]]
        else:
            return ['S', type_tag_list[1]]

    if IOB_tag_list == ['B', 'B', 'O']:
        return ['S', type_tag_list[1]]
    if IOB_tag_list == ['B', 'B', 'I']:
        if type_tag_list[1] == type_tag_list[2]:
            return ['B', type_tag_list[1]]
        else:
            return ['S', type_tag_list[1]]
    if IOB_tag_list == ['B', 'B', 'B']:
        return ['S', type_tag_list[1]]

def caps_vec_map(token):

    assert const.CAPS_DIM == 4
    id_matrix = np.identity(const.CAPS_DIM)

    if token == token.lower():
        caps_vec = id_matrix[0,:]
    elif token == token.upper():
        caps_vec = id_matrix[1,:]
    elif token[0] == token[0].upper():
        caps_vec = id_matrix[2,:]
    else:
        caps_vec = id_matrix[3,:]

    return caps_vec

def normalize_data(data_dict, translation_list, dilation):
    
    # Center and scale the training data:
    # assert 'train' in const.LIST_TRAIN_TESTA_TESTB_KEYS
    # train_data = np.concatenate([data_dict['train'][key_2]['inputs'] for key_2 in data_dict['train']])
    # translation = np.mean(train_data, axis = 0)
    # dilation = np.amax(np.std(train_data - translation, axis = 1))

    # for key_1 in data_dict:
    #     for key_2 in data_dict[key_1]:
            # data_dict[key_1][key_2]['inputs'] -= translation
            # data_dict[key_1][key_2]['inputs'] /= dilation

    data_dict['inputs'] -= np.array(translation_list)
    data_dict['inputs'] /= dilation

    return data_dict

def get_recall_precision_score(labels_1, labels_2):

    prev_tag = ['O', '']
    match_count = 0
    entity_count = 0

    for tag_1, tag_2 in zip(labels_1, labels_2):

        tag_1, tag_2 = tag_1.split('-'), tag_2.split('-')

        # Begin a new tag list:
        if tag_1[0] == 'B':
            tag_1_list = [tag_1]
            tag_2_list = [tag_2]

        elif tag_1[0] == 'I':
            if prev_tag[0] in ['B', 'I'] and prev_tag[1] == tag_1[1]:
                tag_1_list += [tag_1]
                tag_2_list += [tag_2]
            else:
                tag_1_list = []
                tag_2_list = []

        elif tag_1[0] == 'O':
            tag_1_list = []
            tag_2_list = []

        elif tag_1[0] == 'E':
            if prev_tag[0] in ['B', 'I'] and prev_tag[1] == tag_1[1]:
                tag_1_list += [tag_1]
                tag_2_list += [tag_2]
                entity_count += 1
                if list(tag_1_list) == list(tag_2_list):
                    match_count += 1
            tag_1_list = []
            tag_2_list = []

        elif tag_1[0] == 'S':
            entity_count += 1
            if tag_1 == tag_2:
                match_count += 1
            tag_1_list = []
            tag_2_list = []

        prev_tag = tag_1

    if entity_count == 0:
        return False
        
    return match_count / entity_count
