'''
data_handler.py

Written by John Qiu
Date: 2017_06_23

- Takes Thomas's pre-processed token idxs,
- applies cv, train/test/val splits
- for each cv fold
    - init/pckls the following for network initialization
        - wv matrix
        - token to idx mappings
        - label names
    - processes thomas tokens to wv tokens and exports to hdf5 for mpi_learn

#TODO:
    - test pre-trained wv loading
        - implement vector space k checking
'''
import tensorflow as tf
import argparse
import numpy as np
import random
from collections import Counter, defaultdict
import pickle
import h5py
from sklearn import metrics
import os
import shutil
from sklearn.preprocessing import LabelEncoder
import re
import logging

# tf.logging.set_verbosity(tf.logging.INFO)
tf.get_logger().setLevel(logging.INFO)

nontext_nids = ["input_filename", "registryId", "patientIdNumber",
                "tumorRecordNumber", "recordDocumentId"]

parser = argparse.ArgumentParser()

#dir parameters
parser.add_argument("import_dir", type=str,
                    help="output text file generated from epaths_to_vectors")
parser.add_argument("--data_name", type=str, default="big_data",
#parser.add_argument("--data_name", type=str, default="orig_jbhi_epaths",
                    help="output text file generated from epaths_to_vectors")
parser.add_argument("--input_textfile", type=str,
                    default="unify_whitespace_full/record_data_labeled.txt",
                    help="output label file generated from epaths_to_vectors")
parser.add_argument("--input_labelfile", type=str,
                    default="labels/labels_subsite_train.txt",
                    help="output label file generated from epaths_to_vectors")
parser.add_argument("--output_dir", type=str, default="cv_data",
                    help="output directory for hdf5 files")
# data parameters
parser.add_argument('--rand_seed', type=int, default=3545,
                    help='seed for np.rand_state')
parser.add_argument('--min_df', type=int, default=5,
                    help='min token count to initialize unique wv')
parser.add_argument('--pretrained_wv_path', type=str, default=None,
                    help='path for pre-trained WV')
parser.add_argument('--wv_vect_space', type=int, default=300,
                    help='word vector dimensions.')
parser.add_argument('--batch_split', type=int, default=256,
                    help='number of files to split train set into')

parser.add_argument('--pad_size', type=int, default=9,
                    help='length of initial document padding')
parser.add_argument('--seq_len', type=int, default=1500,
                    help='document length data will be truncated/padded to')
parser.add_argument('--reverse_tokens', type=bool, default=True,
                    help='whether to reverse doc tokens')
parser.add_argument('--cv_folds', type=int, default=10,
                    help='number of cv_folds duh what can you not read')
parser.add_argument('--oversample', type=bool,
                    default=False, help='oversample minority classes to majority class')
parser.add_argument('--os_rate', type=float,default=1.0,
                    help='os minority classes to this proprtion of majority class')
parser.add_argument('--val_perc', type=float,
                    default=0.25, help='proportion of train data to used for validation')

def main(args = None):
    '''
    codeflow:
    1) setup:
        - init args, setup dirs, read thomas's files, remove min populated classes
    2) init data processing:
        - numpy seeded shuffle seeded shuffle
        - remove minimally populated class entries
        - (re)encode labels and get label list for balanced cv
    3) fold data processing: for each fold -
        - split train into train/val
        - train set wv processing:
            - iterate through trainset and count vocab
            - attempt to load pre-trained wv
            - initialized wv with count over [min_df] as wv_matrix
            - iterate through trainset and valset assigning wv_matrix idx
        - in dir [output_dir]/[data_name]/[label_file_name]:
            - export label encoder
            - export data_handler_args.txt to summarize handler settings
        - in dir [output_dir]/[data_name]/[label_file_name]/[fold_number]:
             - export data as .hdf5
             - pckl wv_matrix/label_names for network model initilization
             - pckl token_to_mat_idx for future tests ASSUMING SAME PREPROCESSING
    '''

    # 1) setup args, dirs ------------------------------------------------------
    if not args:
        args = parser.parse_args()
    args = parser.parse_args()

    label_file_name = args.input_labelfile
    if "/" in label_file_name:
        label_file_name = label_file_name.split("/")[-1:][0]
    label_file_name = label_file_name.split(".")[0]

    export_dir = os.path.join(args.output_dir,args.data_name,label_file_name)

    print("exporting to", export_dir)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    else:
        print(export_dir, "data dir deleted, will be re-populated")
        shutil.rmtree(export_dir)
        os.makedirs(export_dir)

    in_dir = args.import_dir
    full_textfile_path = os.path.join(in_dir,args.input_textfile)
    full_labelfile_path = os.path.join(in_dir,args.input_labelfile)
    #read thomas processed files
    datalist =  read_vector_text(full_textfile_path, full_labelfile_path)
    #remove minimally populated class (less than 10) cases
    datalist = trim_cases_by_class(datalist)
    #get list of labels
    #seed, split train_test
    rand_state = np.random.RandomState(args.rand_seed)
    #shuffle, get label list, process token_idxlist to numpy array,
    rand_state.shuffle(datalist)
    token_idxlist,labellist = list(zip(*datalist))
    label_encoder = LabelEncoder()
    label_encoder.fit(labellist)
    labellist = label_encoder.transform(labellist)
    token_idxlist = [np.array(x) for x in token_idxlist]
    # zip labellist back into datalist
    datalist = list(zip(token_idxlist,labellist))
    # get cv idx listtoken_to_mat_idx
    cv_idx_list = balancedCV(labellist,args.cv_folds,rand_state)
    # zip datalist to cv_idx_list
    datalist_with_cv = list(zip(datalist,cv_idx_list))
    # get label_file_name, build export_dir, pckl label_encoder

    with open(os.path.join(export_dir,'label_encoder.pickle'), 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(export_dir,'data_handler_args.txt'),'w+') as f:
        f.write(str(args))
    # fold logic start ---------------------------------------------------------
    for this_cv in list(range(args.cv_folds)):
        # split datalist by cv
        test_list = [x[0] for x in datalist_with_cv if x[1]==this_cv]
        train_list = [x[0] for x in datalist_with_cv if x[1]!=this_cv]
        '''
        print(this_cv)
        print("test_list",len(test_list))
        print("train_list",len(train_list))
        '''
        # get trainlist labellist,
        _,labellist = list(zip(*train_list))
        val_list,train_list = list(balanced_split_list(train_list,labellist,args.val_perc))

        # get train_list token count, trim by min_df,
        vocab = get_data_token_count(train_list)
        vocab = {k:v for k,v in vocab.items() if v>=args.min_df}
        # lookup pretrained wv, init pretrained wvs, build token_to_mat_idx mapping
        wv_matrix, token_to_mat_idx = wv_initialize(args.pretrained_wv_path,
                                                    args.min_df,
                                                    vocab,
                                                    rand_state,
                                                    k= args.wv_vect_space)
        train_tokens,train_labels = list(zip(*train_list))
        '''
        token_len = [len(x) for x in train_tokens]
        token_len.sort()
        for i,l in enumerate(token_len):
            print(l,i/len(token_len))
        '''
        # map input tokens to wv_matrix indicies
        # train tokens to wv mappings
        train_idx = [cnn_tokensToIdx(x, token_to_mat_idx,
                                 args.seq_len,
                                 args.pad_size,
                                 args.reverse_tokens) for x in train_tokens]
        train_list = list(zip(train_idx,train_labels))
        # oversample train_list if enabled
        if args.oversample:
            train_list = oversampler(train_list,args.os_rate,rand_state)
        #now tokensToIdx with val
        val_tokens,val_labels = list(zip(*val_list))
        val_idx = [cnn_tokensToIdx(x, token_to_mat_idx,
                               args.seq_len,
                               args.pad_size,
                               args.reverse_tokens) for x in val_tokens]
        val_list = list(zip(val_idx,val_labels))

        #now tokensToIdx with test
        test_tokens,test_labels = list(zip(*test_list))
        test_idx = [cnn_tokensToIdx(x, token_to_mat_idx,
                               args.seq_len,
                               args.pad_size,
                               args.reverse_tokens) for x in test_tokens]
        test_list = list(zip(test_idx,test_labels))

        # split data into x, y
        train_x,train_y = list(zip(*train_list))
        val_x,val_y = list(zip(*val_list))
        test_x,test_y = list(zip(*test_list))
        # set export path to [output_dir]/[data_name]/[label_file_name]/[fold_number]
        export_fold_dir = os.path.join(export_dir,str(this_cv))
        os.makedirs(export_fold_dir)
        # write train/test/val to hdf5
        export_hdf5(train_x,train_y,'train_list',export_fold_dir,batch_split=args.batch_split)
        export_hdf5(val_x,val_y,'val_list',export_fold_dir)
        export_hdf5(test_x,test_y,'test_list',export_fold_dir)
        #pckl wv_matrix, token_to_mat_idx mappings
        label_names = get_list_unique(train_labels)
        with open(os.path.join(export_fold_dir,'label_names.pickle'), 'wb') as handle:
            pickle.dump(label_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(export_fold_dir,'wv_matrix.pickle'), 'wb') as handle:
            pickle.dump(wv_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(export_fold_dir,'token_to_mat_idx.pickle'), 'wb') as handle:
            pickle.dump(token_to_mat_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("wv_matrix, token_to_mat_idx,label_names pckled!")
        print(this_cv,"th fold - case counts:")
        print("test_list", len(test_list))
        print("train_list", len(train_list))
        print("val_list", len(val_list))
    # fold logic end -------------------------------------------------------
    return args

def read_vector_text(input_file, label_file = None):
    #print("Reading input data...")

    # input data is effectively a list of dictionaries
    def parse_record(group):  # convert group of lines into record dict
        item_data = group.split("\n")
        record = [i.strip().split(":") for i in item_data if i != ""]
        return (dict(record))

    data = open(input_file, "r").read().split(";\n")
    # first record is the names of the nIDs mapped to their numbers
    nid2idx = parse_record(data[0])
    idx2nid = dict(((v, k) for k, v in nid2idx.items()))
    records = []
    for group in data[1:]:
        record = parse_record(group)
        # rebuild it with real keys instead of just numbers
        # plus parse from strings (in the metalest way possible)
        record = dict(((idx2nid.get(k, k), eval(v)) for k, v in record.items()))
        records.append(record)

    # load the labels
    # form a dictionary of filename to label idx
    labels = None
    if label_file is not None:
        labels = {}
        f = open(label_file, "r")
        for label in f:
            filename, idx = label.strip().split(",")
            labels[filename] = int(idx)

    #print("Processing records...")
    # now we mush all the different segments together and assign a label
    labeled_records = []  # list of ([data], label)
    for record in records:
        # first, look up label
        label = 0
        if label_file is not None:
            try:
                label = labels[record["input_filename"]]
            except:
                continue  # unlabeled datums are ignored

        # now mash all the segments together based on their alphanum order
        doc = sum([v for k, v in sorted(record.items(), key=lambda r: r[0])
                   if k not in nontext_nids], [])

        labeled_records.append((doc, label))

    return labeled_records

def get_list_unique(in_list):
    '''
    function to remove duplicates in a list
    for python3 transition
    '''
    output = []
    for x in in_list:
        if x not in output:
            output.append(x)
    output.sort()
    return output

def trim_cases_by_class(in_datalist,min_class_freq = 10):
    '''function to take list of [case, label] and returns labels with cases >=
       than min_class_freq
    '''
    label_counter=dict(Counter([x[1] for x in in_datalist]))
    valid_labels = [k for k,v in label_counter.items() if v>= min_class_freq]
    return [x for x in in_datalist if x[1] in valid_labels]

def balancedCV(labelList, numCV, rand_state):
    '''function for class-balanced randomly seeded cross validation
    returns:
        CVlist: list of ints from range(numCV) where idx entry is in test set
    '''
    labelNames = get_list_unique(labelList)
    indexCVList=[]
    for label in labelNames:
        # for each label, get tuple of idx, label tuple then shuffle
        indexListByLabel = [i for i,name in enumerate(labelList) \
                                                 if name == label]
        rand_state.shuffle(indexListByLabel)
        # for shuffled indicies i, assign cv fold with i mod numCV
        indexCVList.extend(list(zip(indexListByLabel,
                            [i % numCV for i,_ in enumerate(indexListByLabel)])))

    # sort by original labelList idx and return just a list of  CV number
    indexCVList.sort(key=lambda x: x[0])
    CVlist = [x[1] for x in indexCVList]

    #error checking
    unqCVlist = get_list_unique(CVlist)
    '''
    print('labelList', len(labelList))
    print('labelNames', len(labelNames))
    print('indexCVList',len(indexCVList))
    print('CVlist',len(CVlist))
    print('unqCVlist',len(unqCVlist))
    print(CVlist)
    '''
    return CVlist


def split_list(inlist,perc):
    """
    function to split list by whatever percent
    """
    split_idx = int(len(inlist)*perc)
    return inlist[:split_idx],inlist[split_idx:]

def balanced_split_list(inlist,in_labels,perc):
    """
    function to split list balanced by indicies of in_labels whatever percent
    """
    out_list_large,out_list_small = [],[]
    for this_label in get_list_unique(in_labels):
        this_label_idx_list = [i for i,x in enumerate(in_labels) if x == this_label]
        large,small = split_list(this_label_idx_list,perc)
        out_list_large.extend([inlist[i] for i in large])
        out_list_small.extend([inlist[i] for i in small])
    return out_list_small,out_list_large
def oversampler(in_datalist,os_rate, in_rand_state):
    '''
    function to oversample input trainlist to class count of max label
    args:
        in_datalist: [(X_train_obs,_label).......]
    returns:
        trainlist: list of oversampled observations
    '''
    trainlist = in_datalist
    label_counter = dict(Counter([x[1] for x in in_datalist]))
    print("total training samples:", len(trainlist))
    print("max_n", max(label_counter.values()), type(max(label_counter.values())))
    max_n = int(max(label_counter.values())*os_rate)
    for this_y in label_counter:
        num_to_os = max_n - label_counter[this_y]
        this_y_list = [x for x in trainlist if x[1] == this_y]
        for i in range(0,int(int(num_to_os)/int(len(this_y_list)))):
            trainlist.extend(this_y_list)
        in_rand_state.shuffle(this_y_list)
        num_os_remainder = num_to_os % len(this_y_list)
        trainlist.extend(this_y_list[:num_os_remainder])
    new_label_counter = dict(Counter([x[1] for x in trainlist]))
    print("Oversampled total training:", len(trainlist))
    return trainlist
'''300
def verify_output(args):
    if not os.path.isdir(args.output):
    try:
        os.mkdir(args.output)
    except Exception as e:
        print("Could not create output dir:", e)
        exit(1)
'''
def parse_record(group): # convert group of lines into record dict
    item_data = group.split("\n")
    record = [i.strip().split(":") for i in item_data if i != ""]
    return(dict(record))

def get_data_token_count(tokenlist):
    """
    function to read training data token token list
    returns:
        vocab: counter of token occurances as ordered dict
    """
    vocab = defaultdict(float)
    #set CV list indicies

    for i, tokens in enumerate(tokenlist):
        vocabCounter = dict(Counter(tokens[0]))
        for this_token in vocabCounter.keys():
            try: vocab[this_token] += 1
            except KeyError: vocab[token] = 1
    return vocab

def wv_initialize(pretrained_path,minDF, train_vocab,rand_state, k=300):
    """reads pre-trained vocab, compares to train_vocab, imports matches,
    randomly initializes others

    """
    # read pre-trained, compare to train_vocab
    wordVecs = {}
    if pretrained_path != None:
        with open(pretrained_path, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                   wordVecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
    #initialize words that aren't read from pre-trained
    for word in train_vocab:
        if word not in wordVecs and train_vocab[word] >= minDF:
            wordVecs[word] = rand_state.uniform(-0.25,0.25,k)
    wordVecs['<unk>'] = rand_state.uniform(-0.25,0.25,k)
    #get WV matrix, idx mappings from orig tokens
    WV_mat, wvToIdx = getidxWVs(wordVecs)

    return WV_mat,wvToIdx

def getidxWVs(loadedWV):
    # Get word matrix, token to wv_mat idx mapping
    vocabSize = len(loadedWV)
    wvToIdx = dict()
    k = 300
    WVmatrix = np.zeros(shape=(vocabSize+1, k), dtype='float32')
    WVmatrix[0] = np.zeros(k, dtype='float32')     #idx for padding is 0
    i = 1
    for word in loadedWV:
        WVmatrix[i] = loadedWV[word]
        wvToIdx[word] = i
        i += 1
    return WVmatrix, wvToIdx

    # B: process pre-trained and initival_listalize untrained wvs
    # load pre-trained WV from pretrained file, and matching tkn vocab
    loadedWV = loadVecs(config['embPath'], vocab)
    # initialize words not in pre-trained with count over minDF
    loadedWV = updateWV(loadedWV, vocab, config['minDF'])
    # get WV matrix and build token to WVmatrix mapping
    WVmat, idxtoWV = getidxWVs(loadedWV)
    return WVmat, idxtoWV

def cnn_tokensToIdx(inTokens, wvToIdx, maxLen , pad_size, reverse_tokens):
    #converts list of tokens into a list of indices with padding - idx 0
    if reverse_tokens:
        inTokens = inTokens[::-1]
    docAsWVidx = []
    padSize = pad_size
    for i in range(padSize):
        docAsWVidx.append(0)
    for word in inTokens:
        if word in wvToIdx and len(docAsWVidx)< maxLen:
            docAsWVidx.append(wvToIdx[word])
        if not word in wvToIdx and len(docAsWVidx)< maxLen:
            docAsWVidx.append(wvToIdx['<unk>'])
    while len(docAsWVidx) < maxLen:
        #pad with zeros
        docAsWVidx.append(0)
    return np.array(docAsWVidx)

def export_hdf5(x_data,y_data,h5_file_name,write_dir,batch_split=None):
    if batch_split== None:
        outfile = h5py.File(os.path.join(write_dir,h5_file_name+".hdf5"), "w")
        outfile.create_dataset("features", data=x_data)
        outfile.create_dataset("labels", data=y_data)
        outfile.close()
        open(os.path.join(write_dir,h5_file_name+".txt"), "w").write(os.path.join(write_dir,h5_file_name+".hdf5\n"))
        print("data written to", os.path.join(write_dir,h5_file_name))
    else:
        print("batch_split",batch_split)
        i = 0
        out_list = open(os.path.join(write_dir,h5_file_name+".txt"), "w+")
        for subx, suby in \
            zip(np.array_split(x_data, batch_split),
                np.array_split(y_data, batch_split)):
            #print("subx",subx)
            #print("suby",suby)
            name = os.path.join(write_dir,h5_file_name+"_{}.hdf5".format(i))
            print(name)
            out_list.write(name+"\n")
            i += 1
            f = h5py.File(name)
            f.create_dataset("features", data=subx)
            f.create_dataset("labels", data=suby)
            f.close()
        out_list.close()

if __name__ == "__main__":
    main()
