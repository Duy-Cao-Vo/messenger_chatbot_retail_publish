import numpy as np
import pandas as pd
import pickle
import random
import json
from underthesea import word_tokenize
from datetime import date
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional
from tensorflow.keras.layers import Embedding, LSTM
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import python_utils
import unidecode
import re

with open('data_input/data_as_an.json', 'r') as f:
    data_as_an = json.load(f)
# print(data_as_an)
# Load df product and store location
df = pd.read_csv('data_input/data_product.csv')
df_address = pd.read_csv('data_input/aldo_location - aldo_store.csv')
df = pd.merge(df, df_address[['store_name', 'address', 'latitude', 'longitude']], on='store_name')

# change Aldo name into ez to read for user
df[['store_index', 'store_name']] = df.store_name.str.split('-', expand=True)
df.loc[~df.store_name.str.contains('Aldo'), 'store_name'] = 'Aldo' + df['store_name']
df = df[~df.store_name.str.contains('Outlet')]

# Split product name into like in fanpage
df[['pro_name', 'code', 'season', 'something1', 'something2']] = df.product_name.str.split(' ', expand=True)
df['pro_name'] = df['pro_name'].str.replace(" ", "")
df['color_code'] = df['color_code'].str.replace(" ", "")
df['pro_name_color'] = df['pro_name'] + " " + df['color_code'].astype('str')

pro_name = df.pro_name.unique().tolist()
pro_name_color = df.pro_name_color.unique().tolist()
color_code = df.color_code.unique().tolist()

# Data Preprocessing to model
pattern_words6 = pro_name  # intent find product code
pattern_words8 = color_code  # intent find color code
negative_emoticons = [':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{']
positive_emoticons = ['=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '😍', '✌', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)', ":)))))))"]


def word2charArr(words, pattern_words_dict):
    words = words.lower()
    arr = []

    # decode icon
    arr.append(18) if any(x in words for x in negative_emoticons) else arr
    arr.append(19) if any(x in words for x in positive_emoticons) else arr

    for i, ch in enumerate(list(words)):
        arr.append(float(ord(ch) + 100))

    # add entities at the end
    word = word_tokenize(words)
    for w in word:
        arr.append((int(pattern_words_dict[str(w)]) + 1)) if w in list(pattern_words_dict.keys()) else arr
        arr.append(6) if w.upper() in pattern_words6 else arr
        arr.append(8) if w.upper() in pattern_words8 else arr
    return arr


def sentence2vec():
    Y_train = []
    X_train = []
    pattern_words_dict = {}
    # loop through each sentence in our intents patterns
    for intent in data_as_an['intents']:
        for entities in intent['entities']:
            if entities:
                pattern_words_dict[entities] = intent['tag'][0:2]
        for sentences in intent['ask']:
            Y_train.append(intent['tag'])
            X_train.append(word2charArr(sentences, pattern_words_dict))
    return np.array(X_train), np.array(Y_train), pattern_words_dict


X_train, Y_train, pattern_words_dict = sentence2vec()
X_train = pad_sequences(X_train, maxlen=100)
highest_unicode = 8100
X_train = np.where(X_train <= highest_unicode, X_train, 0)
print(X_train, X_train.shape, highest_unicode)

from sklearn import preprocessing

cate_enc = preprocessing.LabelEncoder()
label = Y_train
Y_train = cate_enc.fit_transform(Y_train)
print(Y_train.shape)
print(Y_train)
print(len(np.unique(Y_train)))

model = Sequential()
model.add(Embedding(highest_unicode + 1, 60, input_length=X_train.shape[1]))
# model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(Y_train)), activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())


# Fit the model
model.fit(X_train, Y_train, epochs=50, batch_size=24, verbose=1)
# end and save model
model.save('model_intent_classification_lstm_ver2.h5')


def get_random_message(model_return, answer_json):
    sample_responses = []
    for mess in answer_json['intents']:
        if mess['tag'] == model_return:
            sample_responses = mess['answer']
    return random.choice(sample_responses)


# Define pattern word
pattern_words6 = pro_name  # intent find product code
pattern_words8 = color_code  # intent find color code
pattern_words_dict  # pattern_words_dict word needed to emphasize in context data_as_an


def model_predict_intent(sentence):
    data = []
    ERROR_THRESHOLD = 0.25
    print(sentence)
    sentence = sentence.lower()
    data.append(word2charArr(sentence, pattern_words_dict))
    print(data)
    inp = pad_sequences(np.array(data), maxlen=100)
    inp = np.where(inp <= highest_unicode, inp, 0)
    print(inp)
    pred = model.predict(inp)
    prediction = tf.argmax(pred, 1)
    lookup = np.where(Y_train == prediction.numpy()[0])
    context = np.take(label, lookup)[0][0]

    # sort by strength of probability
    results = model.predict(inp)[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    response = get_random_message(context, data_as_an)
    return response, results


# load model LSTM classification
model = tf.keras.models.load_model('model_intent_classification_lstm_ver2.h5')

print(model_predict_intent('Tìm size'))
print(model_predict_intent('Tư vấn size'))
print(model_predict_intent('😍'))
print(model_predict_intent('^^'))
print(model_predict_intent(':('))
print(model_predict_intent("Mình muốn tìm màu 01"))
print(model_predict_intent("mình mới mua cái túi kia, xấu quá, giờ mình muốn đổi thì làm sao"))
print(model_predict_intent("Bạn có người yêu chưa"))
print(model_predict_intent("Hỏi thế gian tình là gì"))
print(model_predict_intent("Bạn ơi cho mình mình hỏi sản phẩm NILIDIEN 97 giá bao nhiêu"))
print(model_predict_intent("Giá sản phẩm"))
print(model_predict_intent("Khuyến mãi"))
print(model_predict_intent("Chương trình khuyến mãi"))
print(model_predict_intent("Tìm một sản phẩm khác"))
