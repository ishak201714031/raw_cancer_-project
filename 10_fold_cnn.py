import json
import os
import re
import pickle
from keras.callbacks import ModelCheckpoint,EarlyStopping
from lib.data_handler import trim_cases_by_class
from lib.data_handler import balanced_split_list
from lib.data_handler import get_data_token_count
from lib.data_handler import wv_initialize
from lib.data_handler import cnn_tokensToIdx
from lib.data_handler import get_list_unique
from lib.data_handler import balancedCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from lib import basic_cnn
from keras.models import load_model
import numpy as np
'''
valid task names:
    gs_behavior_label
    gs_organ_label
    gs_icd_label
    gs_hist_grade_label
    gs_lat_label
'''
# parameters ---------------------------------------------------------------------------
task = 'gs_icd_label'
test_prop = .1
num_cv = 5
val_prop = .25
preloadedWV=None
min_df = 2
pretrained_cnn_name = 'pretrained.h5'
rand_seed = 3545
cnn_seq_len = 1500
reverse_seq = True
train_epochs = 50
def main(args = None):
    rand_state = np.random.RandomState(rand_seed)
    data_label_pairs = get_task_labels(task)
    data_label_pairs = trim_cases_by_class(data_label_pairs)
    label_list = [x[1] for x in data_label_pairs]
    label_encoder = LabelEncoder()
    label_encoder.fit(label_list)
    cv_list = balancedCV(label_list,num_cv,rand_state)

    y_actual,y_pred = [],[]
    for this_cv in range(num_cv):

        # train_idx = [i for i,cv in enumerate(cv_list) if cv != this_cv]
        # test_idx = [i for i,cv in enumerate(cv_list) if cv == this_cv]

        # train = [x for i,x in enumerate(data_label_pairs) if i in train_idx]
        # test = [x for i,x in enumerate(data_label_pairs) if i in test_idx]

        train,test = balanced_split_list(data_label_pairs,label_list,test_prop)
        train_label_list = [x[1] for x in train]
        train,val = balanced_split_list(train,train_label_list,val_prop)
        #get train vocab, initialize train wv matrix, token to wv_idx mappings
        vocab_counter = get_data_token_count(train)

        # Saving the vocab dictionary
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab_counter, f)

        wv_mat, wv_to_idx = wv_initialize(preloadedWV,min_df,vocab_counter,rand_state)

        with open('wv_to_idx.pkl', 'wb') as file:
            pickle.dump(wv_to_idx, file)
            
        with open('wv_mat.npy', 'wb') as handle:
    # Using NumPy's save function since wv_mat is likely a NumPy array
            np.save(handle, wv_mat)
        
        train_tokens,train_y = list(zip(*train))
        train_x = [cnn_tokensToIdx(x,wv_to_idx,cnn_seq_len,0,reverse_seq) \
                   for x in train_tokens]
        val_tokens,val_y = list(zip(*val))
        val_x = [cnn_tokensToIdx(x,wv_to_idx,cnn_seq_len,0,reverse_seq) \
                   for x in val_tokens]
        test_tokens,test_y = list(zip(*test))
        test_x = [cnn_tokensToIdx(x,wv_to_idx,cnn_seq_len,0,reverse_seq) \
                 for x in test_tokens]
        train_y = label_encoder.transform(train_y)
        test_y = label_encoder.transform(test_y)
        val_y = label_encoder.transform(val_y)
        label_names = get_list_unique(train_y)

        #try to load pretrained model, otherwise re-train
        model_name = '_'.join([task,pretrained_cnn_name])

        cnn=basic_cnn.init_full_network(wv_mat,label_names)
        checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
        stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        _ = cnn.fit(x=np.array(train_x),y=np.array(train_y),batch_size=64,epochs=train_epochs,validation_data=tuple((np.array(val_x),np.array(val_y))), callbacks=[checkpointer,stopper])
        top_model = load_model(model_name)
        fold_actual = test_y
        fold_preds_probs = top_model.predict(np.array(test_x))
        fold_preds = [np.argmax(x) for x in fold_preds_probs]

        micro_f = f1_score(fold_actual,fold_preds,average = 'micro')
        macro_f = f1_score(fold_actual,fold_preds,average = 'macro')
        print(this_cv,"fold micro-f", micro_f)
        print(this_cv,"fold macro-f", macro_f)
        y_actual.extend(fold_actual)
        y_pred.extend(fold_preds)

    micro_f = f1_score(y_actual,y_pred,average = 'micro')
    macro_f = f1_score(y_actual,y_pred,average = 'macro')
    print("FULL EXPERIMENT micro-f", micro_f)
    print("FULL EXPERIMENT macro-f", macro_f)

def cleanText(text):
    '''
    function to clean text
    '''
    #replace symbols and tokens
    text = re.sub('\n|\r', ' ', text)
    text = re.sub('o clock', 'oclock', text, flags=re.IGNORECASE)
    text = re.sub(r'(p\.?m\.?)','pm', text, flags=re.IGNORECASE)
    text = re.sub(r'(a\.?m\.?)', 'am', text, flags=re.IGNORECASE)
    text = re.sub(r'(dr\.)', 'dr', text, flags=re.IGNORECASE)
    text = re.sub('\*\*NAME.*[^\]]\]', 'nametoken', text)
    text = re.sub('\*\*DATE.*[^\]]\]', 'datetoken', text)
    text = re.sub("\?|'", '', text)
    text = re.sub('[^\w.;:]|_|-', ' ', text)
    text = re.sub('[0-9]+\.[0-9]+','floattoken', text)
    text = re.sub('floattokencm','floattoken cm', text)
    text = re.sub(' [0-9][0-9][0-9]+ ',' largeint ', text)
    text = re.sub('\.', ' . ', text)
    text = re.sub(':', ' : ', text)
    text = re.sub(';', ' ; ', text)

    #lowercase
    text = text.lower()

    #tokenize
    text = text.split()
    return text

def read_json():
    """
    function to read matched_fd.json as list
    """
    with open('matched_fd.json') as data_file:
        data = json.load(data_file)
    return data

def get_valid_label(task_name,in_data):
    """
    function to get text,labels for valid tasks
    """
    #print(in_data[0])
    valid_entries = [x for x in in_data if x[task_name]['match_status']=="matched"]
    valid_text = [x['doc_raw_text'] for x in valid_entries]
    valid_tokens = [cleanText(x) for x in valid_text]
    valid_labels = [x[task_name]['match_label'] for x in valid_entries]
    return list(zip(valid_tokens,valid_labels))

def get_task_labels(in_task):
    read_data = read_json()
    return get_valid_label(in_task,read_data)


if __name__ == "__main__":
    main()
