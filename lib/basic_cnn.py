"""
Code to export keras architecture/placeholder weights for JBHI CNN
Written by John Qiu
Date: 2017_06_23
"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Embedding
# from keras.layers import merge as Merge
from keras.layers import MaxPooling1D, Convolution1D
# from keras.layers.merge import Concatenate
from tensorflow.keras.layers import Concatenate
from keras import optimizers
import keras.backend as K
import pickle
import argparse
import os
import shutil
from keras.regularizers import l2
parser = argparse.ArgumentParser()

# data/dir parameters
parser.add_argument('--init_full_cvs', type=bool, default=True,
                    help='number of cv_folds duh what can you not read')
parser.add_argument('--model_name', type=str, default="basic_cnn", help='prob of clf layer dims to dropout')
parser.add_argument('--data_name', type=str, default="big_data", help='prob of clf layer dims to dropout')
parser.add_argument('--label_name', type=str, default="labels_subsite_train", help='prob of clf layer dims to dropout')
parser.add_argument('--this_fold', type=str, default=0,
                    help='number of cv_folds duh what can you not read')
# network parameters
parser.add_argument('--rand_seed', type=int, default=3545, help='seed for np.rand_state')
parser.add_argument('--seq_len', type=int, default=1500, help='doc len in tokens')
parser.add_argument('--l2_rate', type=float, default=0.01, help='seed for np.rand_state')
parser.add_argument('--emb_dropout', type=float, default=1.0, help='prob of emb layer dimensions to dropout')
#parser.add_argument('--filter_sizes', type=list, default=[1,2,3,4,5,6,7,8,9,10], help='len of conv filters in paraWV_matrix_pathWV_matrix_pathllel')
parser.add_argument('--filter_sizes', type=list, default=[3,4,5], help='len of conv filters in paraWV_matrix_pathWV_matrix_pathllel')
parser.add_argument('--num_filters', type=int, default=100, help='len of conv-filters(feature maps) for each filter size in parallel')
parser.add_argument('--clf_hidden_dims', type=int, default=100, help='prob of clf layer dims to dropout')
parser.add_argument('--clf_dropout_prob', type=float, default=0.5, help='prob of clf layer dims to dropout')


def init_full_network(wv_matrix,label_names,in_args=None):
    """initializes keras model for basic JBHI cnn and exports architecture as
       json and weights as hdf5. default argparse are paper parameters

    returns:
        model: compiled but untrained keras model
        in_args: to be easily modified for

    codeflow:
    1) setup:
        - from args get pckled wv matrix, label_names
            - input_directory:
                cv_folds/[data_name]/[label_file_name]/[this_fold]/
    2) define network
    3) export structure/weights
            - output_directory:
                models/[model_name]/[data_name]/[label_file_name]/[this_fold]
    """
    if not in_args:
        in_args = parser.parse_args()
    in_args.this_fold = str(in_args.this_fold)
    # set input dir as cv_data/[data_name]/[label_file_name]/[this_fold]
    import_dir = os.path.join('cv_data',
                              in_args.data_name,
                              in_args.label_name,
                              in_args.this_fold)

    # set output dir as models/[model_name]/[data_name]/[label_file_name]/[this_fold]
    output_dir = os.path.join("initialized_models",
                              in_args.model_name,
                              in_args.data_name,
                              in_args.label_name,
                              in_args.this_fold)
    print("exporting to", output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(output_dir, "data dir identified but will be re-populated")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    # unpckl wv_matrix, class_names
    print("valid pre-processed data found in", import_dir)
    # define network layers ----------------------------------------------------
    input_shape = (in_args.seq_len,)
    model_input = Input(shape=input_shape)
    emb_lookup = Embedding(len(wv_matrix), \
                           len(wv_matrix[0]), \
                           input_length=in_args.seq_len, \
                           name="embedding", \
                           embeddings_regularizer=l2(0.0001))(model_input)
    '''
    if in_args.emb_dropout:
        emb_lookup = Dropout(in_args.emb_dropout)(emb_lookup)
    '''
    conv_blocks = []
    for ith_filter,sz in enumerate(in_args.filter_sizes):
        conv = Convolution1D(filters=in_args.num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1,
                             kernel_initializer = 'lecun_uniform',
                             kernel_regularizer=l2(in_args.l2_rate))(emb_lookup)
        #conv_shape = conv.get_shape().as_list()[0]
        conv_shape = in_args.seq_len - sz + 1
        pooled = MaxPooling1D(conv_shape)(conv)
        #pooled = MaxPooling1D(sz, name='{} pool size {}'.format(ith_filter,conv_shape))(conv)
        flat = Flatten()(pooled)
        conv_blocks.append(flat)

    #l_merge = Merge(mode='concat', concat_axis=1)(conv_blocks)
    #concat = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    concat = Concatenate()(conv_blocks)
    #concat = Flatten()(conv_blocks)
    concat_drop = Dropout(in_args.clf_dropout_prob)(concat)
    #z = Dense(in_args.clf_hidden_dims, activation="relu")(z)

    opt = optimizers.Adam()

    model_output = Dense(len(label_names), activation='softmax',
                         kernel_regularizer=l2(in_args.l2_rate))(concat_drop)
    model = Model(model_input, model_output)
    # finished network layers definition - compile network

    # load wv_matrix into embedidng layer
    print("Initializing embedding layer with word2vec weights, shape", wv_matrix.shape)

    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([wv_matrix])
    # end define network layers ----------------------------------------------------
    # save model architecture as json
    #open(os.path.join(output_dir,"structure.json"),"w").write(model.to_json())
    # save initialized model weights as .hdf5
    #model.save_weights(os.path.join(output_dir, "weights"+".hdf5"))
    #print("basic_cnn network/initial weights successfully saved in", output_dir)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["acc"])

    print("cnn settings:")
    print(in_args)
    #print(model.summary())
    return model

def init_half_network(in_pretrained):
    """
    function to load weights of pretrained network to output feedforward until concat layer
    """
    conc_model = Sequential()


if __name__ == "__main__":
    main()
