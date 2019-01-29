#!/usr/bin/env python3

##########################################
#
# Show-and-Tell.py: A Simple and rough implementation of Show-and-Tell with Keras (https://ai.googleblog.com/2016/09/show-and-tell-image-captioning-open.html) 
#                used to generate captions on a given image. Trained on the Flicker8k Dataset (https://forms.illinois.edu/sec/1713398).
#  
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 26/01/2019
#
# This file is distribuited under the terms of GNU General Public
#
########################################


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding,BatchNormalization, Dropout, TimeDistributed, Dense,Concatenate,Merge, RepeatVector, Activation
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from tqdm import tqdm
from keras.preprocessing import image
from keras import backend 
from keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import pickle


# Load all the captions and format them for the LSTM layers, divided in training and test set

# Make caption dictionary whose keys are image file name and values are image caption
token_dir = "Flickr8k_text/Flickr8k.token.txt"

image_captions = open(token_dir).read().split('\n')
caption = {}    
for i in range(len(image_captions)-1):
    id_capt = image_captions[i].split("\t")
    id_capt[0] = id_capt[0][:len(id_capt[0])-2] # get rid of the #0,#1,#2,#3,#4 from the tokens file
    if id_capt[0] in caption:
        caption[id_capt[0]].append(id_capt[1])
    else:
        caption[id_capt[0]] = [id_capt[1]]


# Make two files named "trainImages.txt" and "testImages.txt" that will have start and end token at the start and end of each caption respectively.

train_imgs_id = open("Flickr8k_text/Flickr_8k.trainImages.txt").read().split('\n')[:-1]


train_imgs_captions = open("Flickr8k_text/trainImages.txt",'w')
for img_id in train_imgs_id:
    for captions in caption[img_id]:
        desc = "<start> "+captions+" <end>"
        train_imgs_captions.write(img_id+"\t"+desc+"\n")
        train_imgs_captions.flush()
train_imgs_captions.close()

test_imgs_id = open("Flickr8k_text/Flickr_8k.testImages.txt").read().split('\n')[:-1]

test_imgs_captions = open("Flickr8k_text/testImages.txt",'w')
for img_id in test_imgs_id:
    for captions in caption[img_id]:
        desc = "<start> "+captions+" <end>"
        test_imgs_captions.write(img_id+"\t"+desc+"\n")
        test_imgs_captions.flush()
test_imgs_captions.close()



#----- Use InceptionV3 to get it's predictions for the images_pathand then feed it to the lstm with the captions  ------#

#Load InceptionV3
model = InceptionV3(weights='imagenet')


new_input = model.input
new_output = model.layers[-2].output

model_new = Model(new_input, new_output)

#Pre-process the images_pathto get the predicion tf style from inception

# Scale the pixels between -1 and 1, sample wise. (from keras preprocessing)
def preprocess_input(x):
	x /= 127.5
	x -= 1.
	return x

# Convert all the images_pathinto a numpy array of 3 dimension plus one.
def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x


# Get the prediction from inceptionV3 for the given image
def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc



images_path= 'Flicker8k_Dataset/'

train_imgs_id = open("Flickr8k_text/Flickr_8k.trainImages.txt").read().split('\n')[:-1]
test_imgs_id = open("Flickr8k_text/Flickr_8k.testImages.txt").read().split('\n')[:-1]

# Encode/get prediction for both training and test images

encoding_train = {}
# with tqdm we'll get a nice progress bar for the loops
for img in tqdm(train_imgs_id): 
    path = images_path+str(img)
    encoding_train[img] = encode(path)


with open("encoded_train_images_inceptionV3.p", "wb") as encoded_pickle: 
    pickle.dump(encoding_train, encoded_pickle)  


encoding_train = pickle.load(open('encoded_train_images_inceptionV3.p', 'rb'))

encoding_test = {}
for img in tqdm(test_imgs_id):
    path = images+str(img)
    encoding_test[img] = encode(path)

with open("encoded_test_images_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)


encoding_test = pickle.load(open('encoded_test_images_inceptionV3.p', 'rb'))

#----- Create a bag of words ohe from the caption to be fed to the lstm ----------------------------------------------------#

dataframe = pd.read_csv('Flickr8k_text/trainImages.txt', delimiter='\t')
captionz = []
img_id = []
dataframe = dataframe.sample(frac=1)
iter = dataframe.iterrows()

for i in range(len(dataframe)):
    nextiter = next(iter)
    captionz.append(nextiter[1][1])
    img_id.append(nextiter[1][0])

no_samples=0
tokens = []
tokens = [i.split() for i in captionz]
for caption in captionz:
    no_samples+=len(caption.split())-1


# Load from file it's much faster then recreating the vocab each time.
vocab= [] 
vocab = list(set(vocab))
vocab= pickle.load(open('vocab.p', 'rb'))
vocab_size = len(vocab)

word_idx = {val:index for index, val in enumerate(vocab)}
idx_word = {index:val for index, val in enumerate(vocab)}


caption_length = [len(caption.split()) for caption in captionz]
max_length = max(caption_length)


# Create batches for training
def data_process(batch_size):
    partial_captions = []
    next_words = []
    images_path= []
    total_count = 0
    while 1:
    
        for image_counter, caption in enumerate(captionz):
            current_image = encoding_train[img_id[image_counter]]
    
            for i in range(len(caption.split())-1):
                total_count+=1
                partial = [word_idx[txt] for txt in caption.split()[:i+1]]
                partial_captions.append(partial)
                next = np.zeros(vocab_size)
                #OHE the captions
                next[word_idx[caption.split()[i+1]]] = 1
                next_words.append(next)
                images.append(current_image)

                if total_count>=batch_size:
                    next_words = np.asarray(next_words)
                    images_path= np.asarray(images)
                    partial_captions = sequence.pad_sequences(partial_captions, maxlen=max_length, padding='post')
                    total_count = 0
                
                    yield [[images, partial_captions], next_words]
                    partial_captions = []
                    next_words = []
                    images_path= []


#------ Now we build the encoder-decoder model ----------------------------------------------#

EMBEDDING_DIM = 300 

# Model

image_model = Sequential()
image_model.add(Dense(EMBEDDING_DIM, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_length))

   
lang_model = Sequential()
lang_model.add(Embedding(vocab_size,EMBEDDING_DIM , input_length=max_length))
lang_model.add(Bidirectional(LSTM(256,return_sequences=True)))
lang_model.add(Dropout(0.5))
lang_model.add(BatchNormalization())
lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

   
final_model = Sequential()
final_model.add(Merge([image_model, lang_model], mode='concat'))
#final_model.add(Concatenate([image_model, lang_model]), axis=-1)
final_model.add(Dropout(0.5))
final_model.add(BatchNormalization())
final_model.add(Bidirectional(LSTM(1000,return_sequences=False)))

final_model.add(Dense(vocab_size))
final_model.add(Activation('softmax'))
print ("Model created!")


final_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Training time!!

epoch = 10
batch_size = 128
final_model.fit_generator(data_process(batch_size=batch_size), steps_per_epoch=no_samples/batch_size, epochs=epoch, verbose=1, callbacks=None)

# save the entire model for later use
final_model.save('showandtell.h5')



# Now we see the results after training:

# Standard predictions
def predict_captions(image_file):
    start_word = ["<start>"]
    while 1:
        now_caps = [word_idx[i] for i in start_word]
        now_caps = sequence.pad_sequences([now_caps], maxlen=max_length, padding='post')
        e = encoding_test[image_file]
        preds = final_model.predict([np.array([e]), np.array(now_caps)])
        word_pred = idx_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_length: 
    #keep on predicting next word unitil word predicted is <end> or caption lenghts is greater than max_lenght(40)
            break
            
    return ' '.join(start_word[1:-1])


# Beam search prediction
def beam_search_predictions(image_file, beam_index = 3):
    start = [word_idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            now_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
            e = encoding_test[image_file]
            preds = final_model.predict([np.array([e]), np.array(now_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            #Get the top Beam index = 3  predictions and create a 
            # new list so we can feed them to the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx_word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


# Test the model with an image

image_file ="667626_18933d713e.jpg"
test_image =  images_path+ image_file
Image.open(test_image)

print ('Greedy search:', predict_captions(image_file))
print ('Beam Search, k=3:', beam_search_predictions(image_file, beam_index=3))
print ('Beam Search, k=5:', beam_search_predictions(image_file, beam_index=5))
print ('Beam Search, k=7:', beam_search_predictions(image_file, beam_index=7))

