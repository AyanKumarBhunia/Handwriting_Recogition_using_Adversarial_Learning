#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import time

import tensorflow as tf
import numpy as np
np.random.seed(1234)
import cv2
from matplotlib import pyplot as plt
from natsort import natsorted
from six.moves import xrange as range
import pandas
import os

parameters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

global params
params = {'num_classes' : len(parameters) + 1,
          'wordpath': 'IAMwords.txt',
          'path' : 'IAMhand2printv3',
          'model_path': './model_weights',
          'Batch_Size': 32,
          'lr': 0.0001,
          'num_epochs':1000
}


def get_file_paths(path=params['path']):
    paths = np.array(natsorted([os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files]))
    WordList = np.array([line.rstrip('\r\n') for line in open(params['wordpath'])])
    index = np.random.permutation(len(paths))
    paths = paths[index]
    WordList = WordList[index]
    return paths[:int(len(paths) * .8)], paths[int(len(paths) * .8):], WordList[:int(len(WordList) * .8)], WordList[int(
        len(WordList) * .8):]


def load_img(path):
    img_ = cv2.imread(path)
    img_ = cv2.resize(img_,(100,64))
    #img_ = img_/127.5 - 1.0
    return img_/255.0


def load_weights(saver, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(model_dir, ckpt_name))
        iter_val = int(ckpt_name.split('-')[-1])
        return iter_val, True
    else:
        return False, False


def save(saver, checkpoint_dir, step):
    dir = os.path.join(checkpoint_dir, "model")
    saver.save(sess, dir, step)


keys = [c for c in parameters]
values = [c for c in range(len(parameters))]
letter2num = {}
num2letter = {}

for ik in range(len(parameters)):
    letter2num[keys[ik]] = values[ik]
    num2letter[values[ik]] = keys[ik]


def word2Sparse(labels):
    indices = []
    values = []
    for n, word in enumerate(labels):
        for i, char in enumerate(list(word)):
            indices.append([n, i])
            if (letter2num[char] > 51) or (letter2num[char] < 0):
                print('Error')
                print(word)
            values.append(letter2num[char])

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(labels), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    batch_target = indices, values, shape
    return batch_target  # , Seq_Len

def Sparse2Word(sparse_matrix, Seq_Len):
    start_ = 0
    end_ = 0
    Out_Words = []
    for ik in range(sparse_matrix.dense_shape[0]):
        end_ = end_ + Seq_Len[ik]
        word_sp = sparse_matrix.values[start_: end_]
        start_ = start_ + Seq_Len[ik]
        Out_Words.append(''.join([parameters[i] for i in word_sp]))
    return Out_Words


def Architecture(Input, is_training=True):
    # Input = tf.placeholder(tf.float32, shape = (32, 64, 100, 1))
    is_training = True
    num_classes = params['num_classes']

    with tf.variable_scope("Network"):
        with tf.variable_scope("CNN"):
            conv1 = tf.layers.conv2d(Input, 64, 3, activation=tf.nn.relu, name='conv1', padding='same')
            pool1 = tf.layers.max_pooling2d(conv1, 2, [2, 2], name='pool1')

            conv2 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu, name='conv2', padding='same')
            pool2 = tf.layers.max_pooling2d(conv2, 2, [2, 1], name='pool2', padding='same')

            conv3 = tf.layers.conv2d(pool2, 256, 3, activation=None, name='conv3', padding='same')
            b_norm3 = tf.nn.relu(tf.layers.batch_normalization(conv3, training=is_training, name='batch-norm1'),
                                 name='relu3')

            conv4 = tf.layers.conv2d(b_norm3, 256, 3, activation=tf.nn.relu, name='conv4', padding='same')
            pool4 = tf.layers.max_pooling2d(conv4, 2, [2, 1], name='pool4', padding='same')

            conv5 = tf.layers.conv2d(pool4, 256, 3, activation=tf.nn.relu, name='conv5', padding='same')
            # pool5 = tf.layers.max_pooling2d(conv5, 2,[2,1],name = 'pool5',  padding = 'same')

            conv6 = tf.layers.conv2d(conv5, 256, 3, activation=None, name='conv6', padding='same')
            b_norm6 = tf.nn.relu(tf.layers.batch_normalization(conv6, training=is_training, name='batch-norm2'),
                                 name='relu6')

            conv7 = tf.layers.conv2d(b_norm6, 256, 3, activation=tf.nn.relu, name='conv7', padding='same')
            pool7 = tf.layers.max_pooling2d(conv7, 2, [2, 1], name='pool7', padding='same')

            conv8 = tf.layers.conv2d(pool7, 256, 2, activation=None, name='conv8')
            b_norm8 = tf.nn.relu(tf.layers.batch_normalization(conv8, training=is_training, name='batch-norm3'),
                                 name='relu8')

            shape = b_norm8.get_shape().as_list()
            transposed = tf.transpose(b_norm8, perm=[0, 2, 1, 3], name='transposed')
            conv_reshaped = tf.reshape(transposed, [shape[0], -1, shape[1] * shape[3]], name='reshaped')

    list_n_hidden = [256, 256]
    with tf.name_scope('deep_bidirectional_lstm'):
        fw_cell_list = [tf.contrib.rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        bw_cell_list = [tf.contrib.rnn.BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        lstm_net, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, conv_reshaped,
                                                                        dtype=tf.float32)
        lstm_net = tf.nn.dropout(lstm_net, keep_prob=0.5)  # [width(time), batch, n_classes]
        shape = lstm_net.get_shape().as_list()  # [batch, width, 2*n_hidden]
        rnn_reshaped = tf.reshape(lstm_net, [-1, shape[-1]])  # [batch x width, 2*n_hidden]

        W = tf.Variable(tf.truncated_normal(shape=[512, num_classes], mean=0.0, stddev=0.02))
        b = tf.Variable(tf.constant(value=0.0, shape=[num_classes]))
        fc_out = tf.nn.bias_add(tf.matmul(rnn_reshaped, W), b)

        lstm_out = tf.reshape(fc_out, [shape[0], -1, num_classes], name='reshape_out')
        lstm_out = tf.transpose(lstm_out, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]
        #print(lstm_out)
    return lstm_out


def Compile():
    Input = tf.placeholder(dtype=tf.float32, shape=[params['Batch_Size'], 32, 100, 3], name='Input')
    Target = tf.sparse_placeholder(tf.int32, name='Target')
    Seq_Len = tf.placeholder(dtype=tf.int32, shape=[params['Batch_Size']], name='Seq_len')

    logits = Architecture(Input)
    #print(logits)
    with tf.control_dependencies([tf.less_equal(Target.dense_shape[1],
                                                tf.reduce_max(tf.cast(Seq_Len, tf.int64)))]):
        loss_ctc = tf.reduce_mean(tf.nn.ctc_loss(Target, logits, Seq_Len, ignore_longer_outputs_than_inputs=True))


    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, Seq_Len, merge_repeated=False, beam_width=100, top_paths=2)

    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), Target))

    optimizer = tf.train.MomentumOptimizer(params['lr'], 0.9).minimize(loss_ctc)

    init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    sess.run([init, table_init])

    saver = tf.train.Saver(max_to_keep=5)
    itr, _ = load_weights(saver, params['model_path'])
    if itr == False:
        itr = 0

    tf.summary.scalar('CTC_Loss_Value', loss_ctc)
    tf.summary.scalar('Label_Error_Rate', ler)

    print('Stteing up summary op...')
    summary_op = tf.summary.merge_all()

    print('Setting Up Saver...')
    Train_summary_writer = tf.summary.FileWriter('./log_dir/Train/', sess.graph)
    Test_summary_writer = tf.summary.FileWriter('./log_dir/Test/', sess.graph)

    trainpaths, testpaths, wordTrain, wordTest = get_file_paths()
    Total_Data = len(trainpaths)


    for i in range(params['num_epochs']):  # range(1): #range(params['num_epochs']):

        index = np.random.permutation(Total_Data)
        trainpaths = trainpaths[index]
        wordTrain = wordTrain[index]

        for idx in range(Total_Data // params['Batch_Size']):  # range(1): #

            batch_paths = trainpaths[idx * params['Batch_Size']: (idx + 1) * params['Batch_Size']]
            batch_Words = wordTrain[idx * params['Batch_Size']: (idx + 1) * params['Batch_Size']]
            batch_data = np.array([load_img(path) for path in batch_paths])
            batch_target = word2Sparse(batch_Words)

            batch_SeqL = np.ones(params['Batch_Size']) * 49
            
         

            feed_dict = {Input: batch_data[:, :32, :, :], Target: batch_target, Seq_Len: batch_SeqL}

            _, Train_loss_ctc, Train_summary_str, Train_decoded, Train_ler = sess.run(
                [optimizer, loss_ctc, summary_op, decoded, ler], feed_dict)

            Train_summary_writer.add_summary(Train_summary_str, itr)
            itr = itr + 1
            #print('Epoch:' + str(i) + ' Step:' + str(idx) + ' Iter:' + str(itr) +' train_CTC_loss:' +
             #     str(Train_loss_ctc) + ' Label_Error_Rate:' + str(Train_ler))

            if idx % 100 == 0:
                print('Epoch:' + str(i) + ' Step:' + str(itr) + ' train_CTC_loss:' +
                      str(Train_loss_ctc) + ' Label_Error_Rate:' + str(Train_ler))
                Predicted_Words = Sparse2Word(Train_decoded[0], np.bincount(Train_decoded[0].indices[:, 0],
                                                                            minlength=params['Batch_Size']))
                #print('Done!')
                print(pandas.DataFrame([batch_Words[:1], Predicted_Words[:1]], ['Label', 'Predicted']))

            if itr % 1000 == 0:
                save(saver, params['model_path'], itr)
                print('Model Saved!!')
                
                Total_test_Data = len(testpaths)
                
                tes_accu = []
                tes_ctc = []
                tes_ler = []
                
                for itdx in range(Total_test_Data // params['Batch_Size']):
                    
                    
                    
                    
                    batch_paths = testpaths[itdx * params['Batch_Size']: (itdx + 1) * params['Batch_Size']]
                    
                    batch_Words = wordTest[itdx * params['Batch_Size']: (itdx + 1) * params['Batch_Size']]
                    batch_data = np.array([load_img(path) for path in batch_paths])
                    
                    batch_target = word2Sparse(batch_Words)

                    batch_SeqL = np.ones(params['Batch_Size']) * 49

                    feed_dict = {Input: batch_data[:, :32, :, :], Target: batch_target, Seq_Len: batch_SeqL}

                    Test_loss_ctc, Test_summary_str, Test_decoded, Test_ler = sess.run(
                        [loss_ctc, summary_op, decoded, ler], feed_dict)
                    
    
                    Predicted_Words = Sparse2Word(Test_decoded[0], np.bincount(Test_decoded[0].indices[:, 0],
                                                                            minlength=params['Batch_Size']))

            
                    res  = np.array([i==j for i,j in zip(batch_Words, Predicted_Words)]).astype(np.int8)
                    accuarcy = np.mean(res)
                    
                    tes_accu.append(accuarcy)
                    tes_ctc.append(np.mean(Test_loss_ctc))
                    tes_ler.append(np.mean(Test_ler))
                
                print('### Testing Results:  Epoch:' + str(i) + ' Step:' + str(itr) + ' Test_CTC_loss:' +
                      str(np.mean(tes_ctc)) + ' Label_Error_Rate:' + str(np.mean(tes_ler)) + ' Test Acuracy: '+str(np.mean(tes_accu)))
                    
                    



def get_graph(model_dir = './model_weights/'):
    ckpt_file_path  = model_dir + [i for i in  os.listdir(model_dir) if i.endswith('meta')][0]
    ckpt_file_path_meta = tf.train.latest_checkpoint(model_dir) + '.meta'
    loader = tf.train.import_meta_graph(ckpt_file_path_meta)
    loader.restore(sess, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()
    return graph

def Test(graph):
    Input = graph.get_tensor_by_name('Input:0')
    Target = graph.get_tensor_by_name('Target/indices:0'), graph.get_tensor_by_name('Target/values:0'), graph.get_tensor_by_name('Target/shape:0')
    Seq_Len = graph.get_tensor_by_name('Seq_len:0')
    
    logits = graph.get_tensor_by_name('deep_bidirectional_lstm/transpose_time_major:0')
    #CTCBeamSearchDecoder = graph.get_operation_by_name('Results/CTCBeamSearchDecoder')
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, Seq_Len, merge_repeated=True, beam_width=100, top_paths=2)


    trainpaths, testpaths, wordTrain, wordTest = get_file_paths()
    Total_Data = len(testpaths)
    index = np.random.permutation(Total_Data)
    testpaths = testpaths[index]
    wordTest = wordTest[index]


    for idx in range(Total_Data // params['Batch_Size']):  #
        batch_paths = testpaths[idx * params['Batch_Size']: (idx + 1) * params['Batch_Size']]
        batch_Words = wordTest[idx * params['Batch_Size']: (idx + 1) * params['Batch_Size']]
        batch_data = np.array([load_img(path) for path in batch_paths])
        print (len(batch_Words))
        batch_target = word2Sparse(batch_Words)

        batch_SeqL = np.ones(params['Batch_Size']) * 49

        #feed_dict = {Input: batch_data[:, :32, :, :], Target: batch_target, Seq_Len: batch_SeqL}
        feed_dict = {Input: batch_data[:, :32, :, :], Seq_Len: batch_SeqL}

        _logits, Test_decoded = sess.run([logits, decoded], feed_dict)
        Predicted_Words = Sparse2Word(Test_decoded[0], np.bincount(Test_decoded[0].indices[:, 0],minlength=params['Batch_Size']))
        
        print('Done!')
        print(pandas.DataFrame([batch_Words[:1], Predicted_Words[:1]], ['Label', 'Predicted']))




'''TRAINING'''
#'''
from tensorflow.python.framework import ops
ops.reset_default_graph()
global sess

config = tf.ConfigProto()
sess = tf.Session(config = config)
graph = tf.get_default_graph()

Compile()
#'''

'''TETSING'''
'''
from tensorflow.python.framework import ops
ops.reset_default_graph()

global graph
global sess
sess = tf.Session()

sess.run(tf.global_variables_initializer())
graph= get_graph()
Test(graph)
graph
'''
