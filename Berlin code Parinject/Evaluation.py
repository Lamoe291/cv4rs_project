# -*- coding: utf-8 -*-

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# load doc into memory
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
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

# evaluate the skill of the model
# def evaluate_model(model, descriptions, photos, tokenizer, max_length):
# 	actual, predicted = list(), list()
# 	# step over the whole set
# 	for key, desc_list in descriptions.items():
# 		# generate description
# 		yhat = generate_desc(model, tokenizer, photos[key], max_length)
# 		# store actual and predicted
# 		references = [d.split() for d in desc_list]
# 		actual.append(references)
# 		predicted.append(yhat.split())
# 	# calculate BLEU score
# 	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
# 	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
# 	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
# 	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
        for s in actual:
            for w in s:
                for i in w:
                    if i == 'startseq':
                        w.remove(i)
                    elif i == 'endseq':
                        w.remove(i)

        for s in predicted:
            for w in s:
                if w == 'startseq':
                    s.remove(w)

                elif w == 'endseq':
                    s.remove(w)# calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


filename = 'Clean Descriptions/filenames_trainn.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('Clean Descriptions/descriptions_UCM.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length

max_length =max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = 'Clean Descriptions/filenames_test.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('Clean Descriptions/descriptions_UCM.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features/features_UCM_VGG16.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'model save/adam_l_0_0001_LSTM_256_UAV_Densenet_real-ep038-loss0.832-val_loss1.041.v3.h5' #Put the saved model after training
model = load_model(filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)


##############the following code is to generate the descriptions

import time
list_desc = list()
generated_description_dict = dict()
time_start = time.time()

for keys in test_features:
    
    description = generate_desc(model, tokenizer, test_features[keys], max_length)
    generated_description_dict[keys] =description
#     list_desc.append((keys,description[9:-7]))
    
#     print (description[9:-7])
#     print (test_features[keys])
end_time =time.time()-time_start
print (end_time)
generated_description_dict['494']
generated_description_dict['1198']
generated_description_dict['494']
generated_description_dict['1198']

"saving all the generated descriptions in txt file. this is then needed to evaluate the captions using different metrics"

'''
import time
list_desc = list()
generated_description_dict = dict()
time_start = time.time()

for keys in test_features:
    
    description = generate_desc(model, tokenizer, test_features[keys], max_length)
    generated_description_dict[keys] =description
#     list_desc.append((keys,description[9:-7]))
    
#     print (description[9:-7])
#     print (test_features[keys])
end_time =time.time()-time_start
print (end_time)




def tex_pre(photo_dict, descriptions, predicted_desciptions):
    actual, predicted = list(), list()
    actual_joined, predicted_join = list(), list()
    
    # step over the whole set
    for key, desc_list in descriptions.items():
        # generate description
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
#         print ()
        predicted.append(predicted_desciptions[key].split())
#         print(predicted)
#         print(predicted)
        
        
    for s in actual:
        for w in s:
            for i in w:
                if i == 'startseq':
                    w.remove(i)
                elif i == 'endseq':
                    w.remove(i)

    for s in predicted:
        for w in s:        
            if w == 'startseq':
                s.remove(w)

            elif w == 'endseq':
                s.remove(w)
                
#     print (actual)
#     print (' '.join(predicted[0]))
    for i in range(len(predicted)):
        
        predicted_join.append(' '.join(predicted[i]))
# #         print (predicted_join[i])
    print (len(actual))   
    for k in range(len(actual)):
        list_join = list()
        for j in range(len(actual[k])):
            print (' '.join(actual[k][j]))
            list_join.append(' '.join(actual[k][j]))
        actual_joined.append(list_join)
        
        
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3333333, 0.3333333, 0.3333333, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    print (len(actual_joined))
    return actual_joined, predicted_join
#     print ('METEOR: %f' % meteor_score(actual_joined,predicted_join) )
    
print ('****evaluation test****')
# print ('cc',cc)
a , p = evaluation_test = tex_pre(test_features,test_descriptions,generated_description_dict)
    
referenc_1 = list()
referenc_2 = list()
referenc_3 = list()
referenc_4 = list()
referenc_5 = list()
for i in range(len(a)):
    referenc_1.append(a[i][0])
    referenc_2.append(a[i][1])
    referenc_3.append(a[i][2])
    referenc_4.append(a[i][3])
    referenc_5.append(a[i][4])
print (referenc_1[1])
print (referenc_2[1])
print (referenc_3[1])
print (referenc_4[1])
print (referenc_5[1])



path2 = 'C:/Users/Genci/Documents/Captioning datasets/jupyter notebook syd and ucm/nlg-eval/nlgeval/tests/examples'

with open(path2 +'/referenc_1_RSICD.txt', 'w') as f:
    for i in range(len(referenc_1)):
        f.write(referenc_1[i])
        
        f.write('\n')

with open(path2 +'/referenc_2_RSICD.txt', 'w') as f:
    for i in range(len(referenc_2)):
        f.write(referenc_2[i])
        
        f.write('\n')

with open(path2 +'/referenc_3_RSICD.txt', 'w') as f:
    for i in range(len(referenc_3)):
        f.write(referenc_3[i])
        
        f.write('\n')

with open(path2 +'/referenc_4_RSICD.txt', 'w') as f:
    for i in range(len(referenc_4)):
        f.write(referenc_4[i])
        
        f.write('\n')

with open(path2 +'/referenc_5_RSICD.txt', 'w') as f:
    for i in range(len(referenc_5)):
        f.write(referenc_5[i])
        
        f.write('\n')
        
with open(path2 +'/merge__decayGRU__DECODER_RSICD_TGRS.txt', 'w') as f:
    for i in range(len(p)):
        f.write(p[i])
        
        f.write('\n')

'''