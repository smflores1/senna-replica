import os
import json
import time
import datetime
import numpy as np
import tensorflow as tf
import constants as const
import utils.utils as utils
from sklearn.metrics import accuracy_score

# TO DO: add word2vec model dimension to the wdvec_details.json file.
# TO DO: add caps dimension (4) to the wdvec_details.json file.
# TO DO: replace self.output_shape_list[0] with a num_labels attribute.

class SennaNER(object):

    # Class setter:
    def __setattr__(self, attr_str, value):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        # We reinitialize the class if any of these attributes are directly changed:
        if attr_str in []:
            object.__setattr__(self, attr_str, value)
            self.__init__()
            
        # Otherwise, we do not allow the user to (directly) update the attributes:
        else:
            raise NotImplementedError

    # # Class getter:
    # def __getattr__(self, attr_str):

    #     '''
        
    #     Description:

    #     Inputs:

    #     Output:

    #     '''

    #     return object.__getattr__(self, attr_str)

    # Class deleter:
    def __delattr__(self, attr_str):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        # We don't allow (direct) deletion of attributes:
        if attr_str in []:
            object.__delattr__(self, attr_str)
            
        else:
            raise NotImplementedError

    # Class initializer:
    def __init__(self, model_details_dict):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # The user must specify a project name, a data name, and a model name:
        for key in ['project_name', 'wdvec_name', 'model_name']:
            assert key in model_details_dict
            assert isinstance(model_details_dict[key], str)

        # Include any of these missing key, value pairs in 'model_details_dict':
        if 'learning_rate' not in model_details_dict:
            model_details_dict['learning_rate'] = 0.01
        if 'num_trained_epochs' not in model_details_dict:
            model_details_dict['num_trained_epochs'] = 0

        # Set the inputs to class attributes:
        for key in model_details_dict:
            object.__setattr__(self, key, model_details_dict[key])

        # Set a timestamp as a class attribute:
        object.__setattr__(self, 'timestamp', datetime.datetime.now().isoformat().replace(':', '-'))
        
        # Set the wdvec and model details path and its parent directory paths as class attributes:
        self.set_wdvec_details_path()
        self.set_model_details_path()

        # Load the wdvec details if the necessary file exists:
        if os.path.exists(self.wdvec_details_path):
            self.load_wdvec_details()

        # Otherwise, raise an Exception:
        else:
            raise Exception

        # Load the model details if the necessary file exists:
        if os.path.exists(self.model_details_path):
            self.load_model_details()

        # Otherwise, make a new directory to store the model details and dump them there:
        else:
            self.make_model_path()
            # self.save_model_details() # BRING BACK

        # Initialize a new empty graph:
        tf.reset_default_graph()
        object.__setattr__(self, 'graph', tf.Graph())

        # Fill the graph:
        with self.graph.as_default():
        
            # Inputs placeholder:
            object.__setattr__(self, 'inputs_shape_list', [(self.embedding_dim + self.caps_dim) \
                                                            * (2 * self.padding + 1)])
            inputs = tf.placeholder(name = 'inputs',
                                    dtype = self.train_inputs_datatype,
                                    shape = [None] + self.inputs_shape_list)
            object.__setattr__(self, 'inputs', inputs)

            # Target placeholder:
            object.__setattr__(self, 'output_shape_list', self.nn_design_list[-1]['output_shape'])
            target = tf.placeholder(name = 'target',
                                    dtype = self.train_target_datatype,
                                    shape = [None] + self.output_shape_list)
            object.__setattr__(self, 'target', target)

            # Sentence breaks placeholder:
            breaks = tf.placeholder(name = 'breaks',
                                    dtype = tf.int32,
                                    shape = None)
            object.__setattr__(self, 'breaks', breaks)

            # Create the neural network:
            with tf.variable_scope('neural_network', reuse = tf.AUTO_REUSE):
                object.__setattr__(self, 'inputs_shape', tf.shape(inputs, name = 'inputs_shape'))
                object.__setattr__(self, 'layer_shape_list', [self.inputs_shape])
                self._create_network()
            output = tf.identity(self.output, name = 'output')

            # Create the loss function:
            with tf.variable_scope('loss_function', reuse = tf.AUTO_REUSE):
                self._compute_losses()
            loss = tf.identity(self.loss, name = 'loss')

            # Create the optimizer:
            with tf.variable_scope('optimizer', reuse = tf.AUTO_REUSE):
                optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
            object.__setattr__(self, 'optimizer', optimizer)

            # Create the variable initializer:
            init = tf.global_variables_initializer()
            object.__setattr__(self, 'init', init)

            # Create the saver:
            saver = tf.train.Saver()
            object.__setattr__(self, 'saver', saver)

            # REMOVE:
            object.__setattr__(self, 'var_list', [v for v in tf.global_variables()])

    def __enter__(self):

        # Launch the session:
        sess = tf.Session(graph = self.graph)
        object.__setattr__(self, 'sess', sess)
        
        # Run the variable initializer:
        self.sess.run(self.init)

        # Try to load a previously trained model for either inference or future training:
        checkpoint_file_name = os.path.join(self.model_path, 'checkpoint')
        if os.path.exists(checkpoint_file_name):
            try:
                self.load_model()
            except:
                raise Exception('Failed to restore model at {}.'.format(self.model_path))

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        # Close the session:
        self.sess.close()
        
        # Reset the graph:
        tf.reset_default_graph()

    def _create_network(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        layer = self.inputs

        # Create each layer as specified in 'nn_design_list':
        for i, layer_dict in enumerate(self.nn_design_list):

            # Compute the i:th layer output:
            layer = self._create_layer(layer, layer_dict, tag = str(i + 1))

            # Set the i:th layer shape to a class attribute:
            object.__setattr__(self,
                               'layer_shape_list',
                               self.layer_shape_list + [tf.shape(layer,
                                                                 name = 'layer_' + str(i + 1) + '_output_shape')])


        object.__setattr__(self, 'output', layer)

    def _create_layer(self, inputs, layer_dict, tag):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        layer_type = layer_dict['layer_type']
        with tf.variable_scope('layer_' + tag + '_'+ layer_type, reuse = tf.AUTO_REUSE):
        
            # Fully connected layer:
            if layer_type == 'full_cn':
                output = self._create_full_cn_layer(inputs, layer_dict, tag)

            else:
                # Only fully connected layers are supported right now:
                raise Exception('Layer type {} must be, but is not, "full_cn".'.format(layer_type))

        return output

    def _create_full_cn_layer(self, inputs, layer_dict, tag):

        '''

        Description:

        Inputs:

        Output:

        '''
        
        # Because this layer is fully connected, the input tensor must have rank 2:
        self.check_tensor_rank(inputs, 2)
        
        # Get the activation function:
        activation = self._get_activation(layer_dict)
        
        # Get the shape of the weights matrix and biases vector:
        weight_shape = inputs.get_shape().as_list()[1:] + layer_dict['output_shape']
        biases_shape = layer_dict['output_shape']

        # Initialize the weights and biases using the 'fan-in' method:
        with tf.variable_scope('variable_init', reuse = tf.AUTO_REUSE):
            stddev = tf.sqrt(2 / tf.cast(weight_shape[0], dtype = tf.float64))
            weight_init = tf.truncated_normal(weight_shape, stddev = stddev, dtype = tf.float64)
            biases_init = tf.zeros(biases_shape, dtype = tf.float64)

        # Create the weights and biases variables:
        weight = tf.get_variable(name  = 'weight_' + tag,
                                 dtype = tf.float64,
                                 initializer = weight_init)
        biases = tf.get_variable(name  = 'biases_' + tag,
                                 dtype = tf.float64,
                                 initializer = biases_init)

        # Layer operations:
        output = tf.matmul(inputs, weight, name = 'weight_mul_' + tag)
        output = tf.add(output, biases,    name = 'biases_add_' + tag)
        output = activation(output,        name = 'activation_' + tag)
            
        return output

    def _get_activation(self, layer_dict):

        '''
        
        Description:

        Inputs:

        Output:

        '''
    
        if 'activation' not in layer_dict:
            return tf.identity

        elif layer_dict['activation'] == 'relu':
            return tf.nn.relu

        elif layer_dict['activation'] == 'sigmoid':
            return tf.nn.sigmoid

        elif layer_dict['activation'] == 'identity':
            return tf.identity
        
        else:
            raise Exception('Activation must be one of "relu", "sigmoid", or "identity".')

    def _compute_losses(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # Compute the word-level loss (vanilla cross-entropy):
        if self.loss_type == 'word_level':

            # Cross-entropy loss:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.output,
                                                           labels = self.target)
            # Averge over all tokens to get the final loss:
            average_loss = tf.reduce_mean(loss, name = 'mean_over_inputs')

            # Set the loss as a class attribute:
            object.__setattr__(self, 'loss', average_loss)

            return

        # Compute the sentence-level loss:
        if self.loss_type != 'sentence_level':
            raise Exception('Loss type must be either "word_level" or "sentence_level".')

        with tf.variable_scope('transition_weights', reuse = tf.AUTO_REUSE):

            # Initializer for the transition weights:
            with tf.variable_scope('variable_init', reuse = tf.AUTO_REUSE):
                start_init = tf.truncated_normal([self.output_shape_list[0]], dtype = self.train_inputs_datatype)
                trans_init = tf.truncated_normal([self.output_shape_list[0]] * 2, dtype = self.train_inputs_datatype)
                final_init = tf.truncated_normal([self.output_shape_list[0]], dtype = self.train_inputs_datatype)

            # Transition weights for IOBES tags at the sentence beginning:
            start_weights = tf.get_variable(name  = 'start_weights',
                                            dtype = self.train_inputs_datatype,
                                            initializer = start_init)

            # Transition scores for IOBES tags in the sentence middle:
            trans_weights = tf.get_variable(name  = 'trans_weights',
                                            dtype = self.train_inputs_datatype,
                                            initializer = trans_init)

            # Transition scores for IOBES tags in the sentence end:
            final_weights = tf.get_variable(name  = 'final_weights',
                                            dtype = self.train_inputs_datatype,
                                            initializer = final_init)

        cap_1 = tf.shape(self.breaks)[0] - 1
        
        def body_1(i, loss):
        
            begin = [self.breaks[i], 0]
            size = [self.breaks[i + 1] - self.breaks[i], self.output_shape_list[0]]
            sentence_logits = tf.slice(self.output, begin, size)

            # Send one-hot-encoded true labels to categorical true labels:
            sentence_labels = tf.slice(self.target, begin, size)
            true_labels = tf.argmax(sentence_labels, axis = 1)

            cap_2 = tf.shape(sentence_logits)[0]

            def body_2(j, sentence_delta):
                sentence_delta = tf.reshape(sentence_delta, [-1, 1])
                sentence_delta_array = tf.concat([sentence_delta] * self.output_shape_list[0], axis = 1)
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

        start_loss = tf.zeros(shape = 1, dtype = self.train_inputs_datatype)
        i, loss = tf.while_loop(cond_1, body_1, [0, start_loss])
        
        loss = tf.divide(loss, tf.cast(cap_1, dtype = self.train_inputs_datatype))[0]

        # Set the loss as a class attribute:
        object.__setattr__(self, 'loss', loss)

    def _get_tensor_value(self, feed_dict, output_tensor):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        return self.sess.run(output_tensor, feed_dict = feed_dict)

    def _build_feed_dict(self, input_data, sample = False):

        if not sample:
            sample = list(input_data.keys())

        sentence_break = 0

        feed_dict = {self.inputs: [],
                     self.target: [],
                     self.breaks: [sentence_break]}

        # sample is a list of sentence keys:
        for key in sample:

            sentence_break += len(input_data[key]['inputs'])

            feed_dict[self.inputs] += [input_data[key]['inputs']]
            feed_dict[self.target] += [input_data[key]['target']]
            feed_dict[self.breaks] += [sentence_break]

        feed_dict[self.inputs] = np.concatenate(feed_dict[self.inputs])
        feed_dict[self.target] = np.concatenate(feed_dict[self.target])

        return feed_dict

    def update_network(self, feed_dict):

        '''
        
        Description:

        Inputs:

        Output:

        '''
        
        _, loss = self._get_tensor_value(feed_dict, (self.optimizer, self.loss))
        
        return loss
    
    def train_model(self,
                    # input_data,
                    batch_size = 100,
                    display_step = 1,
                    num_training_epochs = 50):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # TO DO: keep input data as an optional parameter to bypass.

        train_start = time.time()
        print_start = time.time()
        epoch_begin = self.num_trained_epochs

        self.load_wdvec()

        # # print([x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
        # # print([n.name for n in self.graph.as_graph_def().node])
        # # with self.graph.as_default():
        # #     print([v for v in tf.global_variables()])
        # with self.graph.as_default():
        #     with tf.variable_scope('loss_function', reuse = True):
        #         with tf.variable_scope('transition_weights', reuse = True):
        #             print(self.sess.run(tf.get_variable(name = 'start_weights', dtype = tf.float64)))

        # pred_labels = self.predict_labels(self.input_data['train'])
        # pred_labels = np.zeros(self.num_train_datapoints)
        # prob_list = []
        # for tag in const.LIST_KEYWORD_TAGS:
        #     numerator = self.__getattribute__('num_train_' + tag + '_tags')
        #     prob_list += [numerator / self.num_train_datapoints]
        # # pred_labels = np.random.choice(len(prob_list), self.num_train_datapoints, prob_list)
        # pred_labels = [const.LIST_KEYWORD_TAGS[int(i)] for i in pred_labels]

        # eval_feed_dict = {}
        # for key in const.LIST_TRAIN_TESTA_TESTB_KEYS:
        #     eval_feed_dict[key] = self._build_feed_dict(input_data = self.input_data[key])

        # true_labels = self._get_tensor_value(eval_feed_dict['train'], self.target)
        # true_labels = np.argmax(true_labels, axis = 1)
        # true_labels = [const.LIST_KEYWORD_TAGS[int(i)] for i in true_labels]

        # print(len(pred_labels))
        # print(len(true_labels))
        # raise Exception

        prob_list = []
        for tag in const.LIST_KEYWORD_TAGS:
            numerator = self.__getattribute__('num_train_' + tag + '_tags')
            prob_list += [numerator / self.num_train_datapoints]
        
        train_data = self.input_data['train']
        num_samples = len(train_data)

        train_metrics_dict = {}
        for key_1 in const.LIST_TRAIN_TESTA_TESTB_KEYS:
            train_metrics_dict[key_1] = {}
            for key_2 in const.LIST_PERFORMANCE_KEYS:
                train_metrics_dict[key_1][key_2] = {}
                for key_3 in const.LIST_PRED_METHOD_KEYS:
                    train_metrics_dict[key_1][key_2][key_3] = []
        # object.__setattr__(self, 'train_metrics_dict', train_metrics_dict)

        # Create a feed dictionary for the performance metrics:
        eval_feed_dict = {}
        for key in const.LIST_TRAIN_TESTA_TESTB_KEYS:
            eval_feed_dict[key] = self._build_feed_dict(input_data = self.input_data[key])

        for epoch in range(epoch_begin, num_training_epochs):

            epoch_start = time.time()

            avg_loss = 0.
            num_batches_per_epoch = int(num_samples / batch_size)
            
            # Loop over all batches
            for i in range(num_batches_per_epoch):

                # Sample 'batch_size' number of sentences:
                sample = np.random.permutation(list(train_data.keys()))
                sample = list(sample[:batch_size])

                # Create the loss_feed dictionary for this mini-batch:
                loss_feed_dict = self._build_feed_dict(input_data = train_data,
                                                       sample = sample)

                # Fit training using batch data:
                loss = self.update_network(loss_feed_dict)

                # Compute average loss
                avg_loss += loss * (batch_size / num_samples)

            object.__setattr__(self, 'avg_loss', avg_loss)
            object.__setattr__(self, 'num_trained_epochs', self.num_trained_epochs + 1)

            # Write out the performance metrics, weights, and biases:
            if (epoch + 1) % display_step == 0:
                for key_1 in const.LIST_TRAIN_TESTA_TESTB_KEYS:

                    # The cumulative number of mini-batches over which we have trained so far:
                    mini_batch_num = epoch * num_batches_per_epoch + i

                    # Compute the loss:
                    loss = self._get_tensor_value(eval_feed_dict[key_1], self.loss)
                    train_metrics_dict[key_1]['loss']['true'] += [(mini_batch_num, loss)]

                    # # Get the output scores for prediction later:
                    # output = self._get_tensor_value(eval_feed_dict[key_1], self.output)

                    # Get the one-hot-encoded true labels and convert to IOBES lables:
                    true_labels = self._get_tensor_value(eval_feed_dict[key_1], self.target)
                    true_labels = np.argmax(true_labels, axis = 1)
                    true_labels = [const.LIST_KEYWORD_TAGS[int(i)] for i in true_labels]

                    num_datapoints = self.__getattribute__('num_' + key_1 + '_datapoints')

                    # Get the predicted labels and compute accuracy, precision, recall, and F1 score:
                    for key_3 in const.LIST_PRED_METHOD_KEYS:

                        # Get the 'true' predicted labels:
                        if key_3 == 'true':
                            pred_labels = self.predict_labels(self.input_data[key_1])

                        # Get the 'dumb' predicted labels:
                        elif key_3 == 'dumb':
                            pred_labels = np.zeros(num_datapoints)

                        # Get the 'rand' predicted labels:
                        elif key_3 == 'rand':
                            pred_labels = np.random.choice(len(prob_list), num_datapoints, prob_list)

                        # Convert the predicted labels to IOBES lables:
                        pred_labels = [const.LIST_KEYWORD_TAGS[int(i)] for i in pred_labels]
                        
                        # Compute the accuracy:
                        accuracy = accuracy_score(true_labels, pred_labels)
                        train_metrics_dict[key_1]['accuracy'][key_3] += [(mini_batch_num, accuracy)]

                        # Compute the precision:
                        precision = utils.get_recall_precision_score(pred_labels, true_labels)
                        if precision:
                            train_metrics_dict[key_1]['precision'][key_3] += [(mini_batch_num, precision)]

                        # Compute the recall:
                        recall = utils.get_recall_precision_score(true_labels, pred_labels)
                        if recall:
                            train_metrics_dict[key_1]['recall'][key_3] += [(mini_batch_num, recall)]

                        # Compute the F1 score:
                        if precision != 0 and recall != 0:
                            f1_score = (2 * precision * recall) / (precision + recall)
                            train_metrics_dict[key_1]['f1_score'][key_3] += [(mini_batch_num, f1_score)]

                # # Output the performance metrics in a json document:
                # utils.save_train_metrics_dict(model_subdir, train_metrics_dict)

                print_out = 'Epoch {} of {} completed.'.format(epoch + 1, num_training_epochs)

                print('\n')
                print(print_out)
                print('-' * len(print_out) * 2)
                print('-' * len(print_out) * 2)
                print('\n')

                print('Time elapsed during training so far    :', round(time.time() - train_start, 3))
                print('Time elapsed since the last checkpoint :', round(time.time() - print_start, 3))
                print('Time elapsed for the last epoc training:', round(time.time() - epoch_start, 3))
                print('\n')

                for key_2 in const.LIST_PERFORMANCE_KEYS:
                    for key_1 in const.LIST_TRAIN_TESTA_TESTB_KEYS:
                        try:
                            print('Cumulative ' + key_1 + ' ' + key_2 + ' over the last epoc:', round(train_metrics_dict[key_1][key_2]['true'][-1][1], 3))
                        except:
                            pass
                    print('\n')

                print_start = time.time()

                self.save_model()

    def predict_labels(self, input_data):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        eval_feed_dict = self._build_feed_dict(input_data)
        output, breaks = self._get_tensor_value(eval_feed_dict, (self.output, self.breaks))
        # print(breaks)

        # Get word-level label predictions:
        if self.loss_type == 'word_level':

            pred_labels = np.argmax(output, axis = 1)

            return pred_labels

        # Otherwise, get sentence-level label predictions:

        with self.graph.as_default(): # For some reason, the graph is not visible without this line.
            with tf.variable_scope('loss_function', reuse = True):
                with tf.variable_scope('transition_weights', reuse = True):
                    frozen_start_weights = self.sess.run(tf.get_variable(name = 'start_weights', dtype = self.train_inputs_datatype))
                    frozen_trans_weights = self.sess.run(tf.get_variable(name = 'trans_weights', dtype = self.train_inputs_datatype))
                    frozen_final_weights = self.sess.run(tf.get_variable(name = 'final_weights', dtype = self.train_inputs_datatype))

        pred_labels = []
        sentence_outputs = np.split(output, breaks[1:-1])

        for sentence_output in sentence_outputs:

            sentence_labels_vecs = []
            num_tokens = sentence_output.shape[0]
            num_labels = sentence_output.shape[1]
            sentence_delta = frozen_start_weights + sentence_output[0]
            
            for i in range(1, num_tokens):

                num_labels = frozen_trans_weights.shape[1]
                sentence_delta_array = np.stack([sentence_delta] * num_labels, axis = 1)
                sentence_delta = np.max(frozen_trans_weights + sentence_delta_array, axis = 0)
                sentence_delta += sentence_output[i]

                sentence_labels_vecs += [np.argmax(frozen_trans_weights + sentence_delta_array, axis = 0)]

            pred_label = np.argmax(sentence_delta + frozen_final_weights)
            sentence_pred_labels = [pred_label]
            for sentence_labels_vec in sentence_labels_vecs[::-1]:
                pred_label = sentence_labels_vec[pred_label]
                sentence_pred_labels = [pred_label] + sentence_pred_labels

            pred_labels += sentence_pred_labels

        pred_labels = np.array(pred_labels)

        return pred_labels

    def set_project_path(self):

        '''
        s
        Description:

        Inputs:

        Output:

        '''

        project_path = os.path.join(const.PATH_ROOT, self.project_name)
        object.__setattr__(self, 'project_path', project_path)

    def set_wdvecs_path(self):

        self.set_project_path()
        wdvec_path = os.path.join(self.project_path, const.DIR_WDVECS)
        object.__setattr__(self, 'wdvecs_path', wdvec_path)

    def set_models_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_project_path()
        models_path = os.path.join(self.project_path, const.DIR_MODELS)
        object.__setattr__(self, 'models_path', models_path)

    def set_wdvec_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_wdvecs_path()
        wdvec_path = os.path.join(self.wdvecs_path, self.wdvec_name)
        object.__setattr__(self, 'wdvec_path', wdvec_path)

    def set_model_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_models_path()
        model_path = os.path.join(self.models_path, self.model_name)
        object.__setattr__(self, 'model_path', model_path)
        saved_model_path = os.path.join(self.model_path, const.DIR_SAVED_MODEL)
        object.__setattr__(self, 'saved_model_path', saved_model_path)

    def set_wdvec_details_path(self):

        self.set_wdvec_path()
        wdvec_details_path = os.path.join(self.wdvec_path, const.WDVEC_DETAILS)
        object.__setattr__(self, 'wdvec_details_path', wdvec_details_path)

    def set_model_details_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_model_path()
        model_details_path = os.path.join(self.model_path, const.MODEL_DETAILS)
        object.__setattr__(self, 'model_details_path', model_details_path)

    def make_project_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_project_path()
        if not os.path.exists(self.project_path):
            os.mkdir(self.project_path)

    def make_wdvecs_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_wdvecs_path()
        if not os.path.exists(self.wdvecs_path):
            self.make_project_path()
            os.mkdir(self.wdvecs_path)

    def make_models_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_models_path()
        if not os.path.exists(self.models_path):
            self.make_project_path()
            os.mkdir(self.models_path)

    def make_model_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_wdvec_path()
        if not os.path.exists(self.wdvec_path):
            self.make_wdvecs_path()
            object.__setattr__(self, 'wdvec_name', self.wdvec_name + '_' + self.timestamp)
            object.__setattr__(self, 'wdvec_path', self.wdvec_path + '_' + self.timestamp)
            os.mkdir(self.wdvec_path)
            self.set_wdvec_details_path()

    def make_model_path(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        self.set_model_path()
        if not os.path.exists(self.model_path):
            self.make_models_path()
            object.__setattr__(self, 'model_name', self.model_name + '_' + self.timestamp)
            object.__setattr__(self, 'model_path', self.model_path + '_' + self.timestamp)
            saved_model_path = os.path.join(self.model_path, 'saved_model')
            object.__setattr__(self, 'saved_model_path', saved_model_path)
            os.mkdir(self.model_path)
            # os.mkdir(self.saved_model_path)
            self.set_model_details_path()

    def load_wdvec_details(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        with open(self.wdvec_details_path) as file_handle:

            loaded_wdvec_details_dict = json.load(file_handle)

            for key in loaded_wdvec_details_dict:
                object.__setattr__(self, key, loaded_wdvec_details_dict[key])
            self.check_wdvec_details_dict_format()

    def check_wdvec_details_dict_format(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        pass

    def load_wdvec(self):

        # Load the data:
        data = {}
        for key_1 in const.LIST_TRAIN_TESTA_TESTB_KEYS: # TO DO: infer keys from data, make in to attr

            file_name = os.path.join(self.wdvec_path, key_1 + '.npy')
            numpy_data = np.load(file_name)
            data[key_1] = numpy_data[()]
        
        # Normalize the data so the training data is centered:
        assert 'train' in const.LIST_TRAIN_TESTA_TESTB_KEYS # TO DO: infer keys from data, make in to attr
        train_data = np.concatenate([data['train'][key_2]['inputs'] for key_2 in data[key_1]])
        translation = np.mean(train_data, axis = 0)
        dilation = np.amax(np.std(train_data - translation, axis = 1))

        for key_1 in const.LIST_TRAIN_TESTA_TESTB_KEYS: # TO DO: infer keys from data, make in to attrs
            for key_2 in data[key_1]:
                data[key_1][key_2]['inputs'] -= translation
                data[key_1][key_2]['inputs'] /= dilation

        object.__setattr__(self, 'input_data', data)

    def load_model(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # Load the weights and checkpoint files:
        assert os.path.exists(self.model_path)
        save_model_path = os.path.join(self.model_path, self.model_path.split('/')[-1])
        self.saver.restore(self.sess, save_model_path)

    def save_model(self):

        '''
        
        Description:

        Inputs:

        Output:

        '''

        # Save the weights and checkpoint files only:
        assert os.path.exists(self.model_path)
        save_model_path = os.path.join(self.model_path, self.model_path.split('/')[-1])
        self.saver.save(self.sess, save_model_path)
        # self.save_model_details() # BRING BACK

    def check_tensor_rank(self, tensor, rank):

        assert len(tensor.get_shape().as_list()) == rank

    def _strip_consts(self, max_const_size = 32):

        '''
        
        Description: Strip large constant values from 'graph_def'.

        Inputs:

        Output:

        '''
        
        strip_def = tf.GraphDef()
        graph_def = self.graph.as_graph_def()
        
        for n0 in graph_def.node:
            
            n = strip_def.node.add() 
            n.MergeFrom(n0)
            
            if n.op == 'Const':
                
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
                    
        return strip_def
    
    def show_graph(self, max_const_size = 32):

        '''
        
        Description: Borrowed from
        https://stackoverflow.com/questions/38189119/simple-way-to-visualize-a-tensorflow-graph-in-jupyter

        Inputs:

        Output:

        '''

        from IPython.display import display, HTML
            
        strip_def = self._strip_consts(max_const_size = max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data = repr(str(strip_def)), id = 'graph' + str(np.random.rand()))

        iframe = """
            <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
        """.format(code.replace('"', '&quot;'))
        display(HTML(iframe))
