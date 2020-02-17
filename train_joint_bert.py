"""
@author: mwahdan
"""

from readers.goo_format_reader import Reader
from vectorizers.bert_vectorizer import BERTVectorizer
from vectorizers.tags_vectorizer import TagsVectorizer
from models.joint_bert import JointBertModel

import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import ModelCheckpoint


def preprocessor(text_arr, tags_arr, intents):
    train_text_arr = np.array(text_arr)
    train_tags_arr = np.array(tags_arr)
    val_text_arr = np.array(text_arr)
    val_tags_arr = np.array(tags_arr)
    train_intents = np.array(intents)
    val_intents = np.array(intents)

    # vectorize data
    bert_vectorizer = BERTVectorizer(sess, is_bert, bert_model_hub_path)
    train_input_ids, train_input_mask, train_segment_ids, train_valid_positions, train_sequence_lengths = bert_vectorizer.transform(
        train_text_arr)
    val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths = bert_vectorizer.transform(
        val_text_arr)

    # vectorize tags
    tags_vectorizer.fit(train_tags_arr)
    train_tags = tags_vectorizer.transform(train_tags_arr, train_valid_positions)
    val_tags = tags_vectorizer.transform(val_tags_arr, val_valid_positions)
    slots_num = len(tags_vectorizer.label_encoder.classes_)

    # encode labels
    train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
    val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
    intents_num = len(intents_label_encoder.classes_)

    return ([train_input_ids, train_input_mask, train_segment_ids, train_valid_positions], [train_tags, train_intents]), \
            ([val_input_ids, val_input_mask, val_segment_ids, val_valid_positions], [val_tags, val_intents]), (slots_num, intents_num)


# read command-line parameters
parser = argparse.ArgumentParser('Training the Joint BERT NLU model')
parser.add_argument('--train', '-t', help='Path to training data in Goo et al format', type=str, required=True)
parser.add_argument('--val', '-v', help='Path to validation data in Goo et al format', type=str, required=True)
parser.add_argument('--save', '-s', help='Folder path to save the trained model', type=str, required=True)
parser.add_argument('--epochs', '-e', help='Number of epochs', type=int, default=5, required=False)
parser.add_argument('--batch', '-bs', help='Batch size', type=int, default=64, required=False)
parser.add_argument('--type', '-tp', help='bert   or    albert', type=str, default='bert', required=False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
train_data_folder_path = args.train
val_data_folder_path = args.val
save_folder_path = args.save
epochs = args.epochs
batch_size = args.batch
type_ = args.type


tf.compat.v1.random.set_random_seed(7)


sess = tf.compat.v1.Session()

if type_ == 'bert':
    bert_model_hub_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))

print('read data ...')
#train_text_arr, train_tags_arr, train_intents = Reader.read(train_data_folder_path)
text_arr, tags_arr, intents = Reader.read(train_data_folder_path)

model = None
#for train_index, val_index in StratifiedKFold(n_splits=5).split(text_arr, tags_arr, intents):
#train_text_arr = np.array(text_arr)
#train_tags_arr = np.array(tags_arr)
#val_text_arr = np.array(text_arr)
#val_tags_arr = np.array(tags_arr)
#train_intents = np.array(intents)
#val_intents = np.array(intents)
#train_text_arr, train_tags_arr, val_text_arr, val_tags_arr,  = train_text_arr[train_index], \
#                                train_tags_arr[train_index], train_text_arr[val_index], train_tags_arr[val_index]
#train_intents, val_intents = train_intents[train_index], val_intents[val_index]

#val_text_arr, val_tags_arr, val_intents = Reader.read(val_data_folder_path)

print('vectorize data ...')
bert_vectorizer = BERTVectorizer(sess, is_bert, bert_model_hub_path)
#train_input_ids, train_input_mask, train_segment_ids, train_valid_positions, train_sequence_lengths = bert_vectorizer.transform(train_text_arr)
input_ids, input_mask, segment_ids, valid_positions, sequence_lengths = bert_vectorizer.transform(text_arr)
#val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths = bert_vectorizer.transform(val_text_arr)


#print('vectorize tags ...')
tags_vectorizer = TagsVectorizer()
#tags_vectorizer.fit(train_tags_arr)
#train_tags = tags_vectorizer.transform(train_tags_arr, train_valid_positions)
tags_vectorizer.fit(tags_arr)
tags = tags_vectorizer.transform(tags_arr, valid_positions)
#val_tags = tags_vectorizer.transform(val_tags_arr, val_valid_positions)
slots_num = len(tags_vectorizer.label_encoder.classes_)


#print('encode labels ...')
intents_label_encoder = LabelEncoder()
#train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
intents = intents_label_encoder.fit_transform(intents).astype(np.int32)
#val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
intents_num = len(intents_label_encoder.classes_)


if model is None:
    model = JointBertModel(slots_num, intents_num, bert_model_hub_path, sess, num_bert_fine_tune_layers=10, is_bert=is_bert)
else:
    model = JointBertModel.load(save_folder_path, sess)

#X = [train_input_ids, train_input_mask, train_segment_ids, train_valid_positions]
#Y = [train_tags, train_intents]

print('training model ...')
#checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
#model.fit([train_input_ids, train_input_mask, train_segment_ids, train_valid_positions], [train_tags, train_intents],
#          validation_data=([val_input_ids, val_input_mask, val_segment_ids, val_valid_positions],
#                           [val_tags, val_intents]), epochs=epochs, batch_size=batch_size,
#          shuffle=True, callbacks=[checkpointer])


print('================================================================================')
print(input_ids.shape, input_mask.shape, segment_ids.shape, valid_positions.shape, tags.shape, intents.shape)
X = np.concatenate((input_ids, input_mask, segment_ids, valid_positions, tags), axis=1)
Y = intents

print('================================================================================')


#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#results = cross_val_score(model, X, Y, cv=kfold, scoring='f1')


model.fit(X, Y, epochs=epochs, batch_size=batch_size)

print('Saving ..')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print('Folder `%s` created' % save_folder_path)
model.save(save_folder_path)
with open(os.path.join(save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
    pickle.dump(tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
    pickle.dump(intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


tf.compat.v1.reset_default_graph()
