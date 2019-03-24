import tensorflow as tf
import constants as const

def loss_function(dtype,
                  logits,
                  labels,
                  breaks,
                  num_labels):

    # # Define the transition weights for IOBES tags at the sentence beginning:
    # start_weights = tf.Variable(tf.truncated_normal([train_details['num_output_units']],
    #                                                  dtype = train_details['train_inputs_datatype']))
    # # Define the transition scores for IOBES tags in the sentence middle:
    # trans_weights = tf.Variable(tf.truncated_normal([train_details['num_output_units']] * 2,
    #                                                  dtype = train_details['train_inputs_datatype']))
    # # Define the transition scores for IOBES tags at the sentence end:
    # final_weights = tf.Variable(tf.truncated_normal([train_details['num_output_units']],
    #                                                  dtype = train_details['train_inputs_datatype']))
    
    cap_1 = tf.shape(breaks)[0] - 1
    
    def body_1(i, loss):
    
        begin = [breaks[i], 0]
        size = [breaks[i + 1] - breaks[i], num_labels]
        sentence_logits = tf.slice(logits, begin, size)

        # Send one-hot-encoded true labels to categorical true labels:
        sentence_labels = tf.slice(labels, begin, size)
        true_labels = tf.argmax(sentence_labels, axis = 1)

        cap_2 = tf.shape(sentence_logits)[0]

        def body_2(j, sentence_delta):
            sentence_delta = tf.reshape(sentence_delta, [-1, 1])
            sentence_delta_array = tf.concat([sentence_delta] * num_labels, axis = 1)
            sentence_delta = tf.reduce_logsumexp(trans_weights + sentence_delta_array, axis = 0)
            sentence_delta += sentence_logits[j]
            return j + 1, sentence_delta

        def cond_2(j, sentence_delta):
            return j < cap_2

        first_delta = start_weights + sentence_logits[0]
        j, sentence_delta = tf.while_loop(cond_2, body_2, [1, first_delta])
        
        def body_3(k, sentence_score):
            sentence_score += trans_weights[true_labels[k - 1], true_labels[k]]
            sentence_score += sentence_logits[k][true_labels[k]]
            return k + 1, sentence_score
        
        def cond_3(k, sentence_score):
            return k < cap_2
        
        first_score = start_weights[true_labels[0]] + sentence_logits[0, true_labels[0]]
        k, sentence_score = tf.while_loop(cond_3, body_3, [1, first_score])
        sentence_score += final_weights[true_labels[k - 1]] # Possible bug, if while_loop doesn't run, did k get a value?
        
        sentence_loss = sentence_score - (tf.reduce_logsumexp(sentence_delta + final_weights))

        return i + 1, loss - sentence_loss

    def cond_1(i, loss):
        return i < cap_1

    start_loss = tf.zeros(shape = 1, dtype = dtype)
    i, loss = tf.while_loop(cond_1, body_1, [0, start_loss])
    
    loss = tf.divide(loss, tf.cast(cap_1, dtype = 'float64'))[0]

return loss

def get_recall_precision_score(labels_1, labels_2):

    prev_tag = ['O', '']
    match_count = 0
    keyword_count = 0

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
                keyword_count += 1
                if list(tag_1_list) == list(tag_2_list):
                    match_count += 1
            tag_1_list = []
            tag_2_list = []

        elif tag_1[0] == 'S':
            keyword_count += 1
            if tag_1 == tag_2:
                match_count += 1
            tag_1_list = []
            tag_2_list = []

        prev_tag = tag_1

    if keyword_count == 0:
        return False
        
    return match_count / keyword_count
