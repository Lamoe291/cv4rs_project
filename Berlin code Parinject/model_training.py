import tensorflow as tf

from numpy import array
from os import listdir
from pickle import dump
from pickle import load
import numpy as np
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import os
import random as rn
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
#tf.random.set_seed(5)
tf.set_random_seed(1111)


def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features

# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = tf.keras.preprocessing.text.Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]

				out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)

# define the captioning model Iputs 1 = image features, Input 2 = word sequence
def define_model(vocab_size, max_length):
    inputs1 = tf.keras.layers.Input(shape=(4096,))
    
    fe1 = tf.keras.layers.Dropout(0.5)(inputs1)
    normalized_cnn_feats = tf.keras.layers.BatchNormalization(axis=-1)(fe1)
    fe2 = tf.keras.layers.Dense(256)(normalized_cnn_feats)
#     print (fe2)
#     normalized_cnn_feats = tf.keras.layers.BatchNormalization(axis=-1)(fe2)
    final_cnn_feats = tf.keras.layers.Masking()(tf.keras.layers.RepeatVector(1)(fe2)) 
#     final_cnn = 
    inputs2 = tf.keras.layers.Input(shape=(max_length,))
    se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = tf.keras.layers.Dropout(0.5)(se1)
    merge = tf.keras.layers.concatenate([final_cnn_feats,se2], axis=1)
    se3 = tf.keras.layers.GRU(256)(merge)

    decoder2 = tf.keras.layers.Dropout(0.5)(se3)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    sgd = tf.keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)
    model.compile(loss='categorical_crossentropy', optimizer=sgd )
    # summarize model
    print(model.summary())
#     tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    return model

# load training dataset (6K)
filename = 'Clean Descriptions/filenames_train.txt'
train = load_set(filename)
print('Dataset train imag: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('Clean Descriptions/descriptions_UCM.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_features('features/features_UCM_VGG16.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
print ('tokenizer',tokenizer)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)
# print (X1train) #features
# print (X2train)# real sequences
# print ('ytrain ',ytrain) #sequence prediction
# dev dataset

# load test set
filename = 'Clean Descriptions/filenames_val.txt'
VALIDATION = load_set(filename)
print('Dataset validation: %d' % len(VALIDATION))
# descriptions
VALIDATION_descriptions = load_clean_descriptions('Clean Descriptions/descriptions_UCM.txt', VALIDATION)
print('Descriptions: validation=%d' % len(VALIDATION_descriptions))
# photo features
test_features = load_photo_features('features/features_UCM_VGG16.pkl', VALIDATION)
print('Photos: validation=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_sequences(tokenizer, max_length, VALIDATION_descriptions, test_features)

# fit model

# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
training_history_filename = 'model_UCM/training_history.log'
csv_logger = tf.keras.callbacks.CSVLogger(training_history_filename, append=False)
filepath = 'model_UCM/par_inject_GRU_conc_default_ADAM-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.v3.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit([X1train, X2train], ytrain, epochs=100, verbose=2, callbacks=[checkpoint,csv_logger], validation_data=([X1test, X2test], ytest),batch_size=128)
#f=K.function([model.layers[0].input, model.layers[1].input],[model.layers[-2].output])
#o = f([X1train[:44708,:], X2train[:44708, :]])[0]

