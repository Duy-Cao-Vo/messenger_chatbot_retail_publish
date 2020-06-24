"""
# Status: DATE: ------ UPDATE
            2020/03/20 UPDATE FIX FIND PRODUCT ID
            2020/03/22 UPDATE SEND SIMPLE RESPONSE AT SESSION 6
            2020/03/24 UPDATE QuickReplies and product Carousel
            2020/03/25 CREATE JSON FILE LOOP TO SAVE USER MESSAGE
                        CAN SEND IMAGE
            2020/03/28 UPDATE CODE TO GITHUB
            2020/03/28 RUN SERVER APPLICATION with HEROKU
            2020/04/11 IMPROVE ENTITIES FUNCTION
            2020/04/12 ADD BUTTIONS POSTBACK
            2020/04/13 ADD 14,15,16,17,18 intent, and TALK TO HUMAN MODE
            2020/04/14 ADD EMOJI MODE
            2020/04/17 ADD PRODUCT PRICE, LOVE, JOKED...
            2020/04/18 ADD COVID-19
            2020/04/28 ADD Function to extract image
            2020/05/05 ADD QUERY MODE
            2020/05/09 ADD PROMOTION MODE
            2020/05/10 ADD Auto marketing started
            maybe add function send email if error
"""

'''
# CODE STRUCTURE: 
        SESSION 1 IMPORT DATA AND MODEL
        SESSION 2 CREATE DATA PREPORCESSING
        SESSION 3 CREATE FUNCTION ACTION FOR CHATBOT
          + Model classification intent and answer
          + Find product code inventory
          + Find size
          + Find the neareast store locate

          ADD QUICK REPLY (Add suggest answers for users)
          ADD PRODUCT CAROUSEL (shoes, bags, accesories)
        SESSION 4 VERIFY TOKEN (KEY TO ACCESS PAGE AND KEY URL)
        
        SESSION 5 MAIN APP
'''
import random
from flask import Flask, request
# from flask_session import Session
import pymongo
from pymongo import MongoClient
# import dns

from pymessenger.bot import Bot

# Thing need for tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Normal LIB
import pandas as pd
import pickle
import random
import json
from datetime import datetime, timedelta, date
import time
import numpy as np
import os

# NLP LIB
# import unidecode
from underthesea import word_tokenize

# Location LIB
from opencage.geocoder import OpenCageGeocode
from math import sin, cos, sqrt, atan2, radians
import requests

# Image LIB
import pytesseract
from PIL import Image

# Web-browser LIB
import pycurl
import certifi
import io as bytesIOModule
import requests
import webbrowser
from bs4 import BeautifulSoup

# do not show Warnings
import warnings

warnings.filterwarnings("ignore")

# import Data Carousel WebElements
import data_input.DataWeb as DataWeb

dataObj = DataWeb.WebElements()
shoes_women_elements = dataObj.shoes_women
bags_elements = dataObj.bags
shoes_men_elements = dataObj.shoes_men

'''
    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------- SESSION 1 IMPORT DATA AND MODEL  ---------------------------------------------
'''
# # CODE TO QUERY DATA FROM BIGQUERY
# from google.cloud import bigquery
# from google.oauth2 import service_account


def Query_data_bq(table_name: str, query_mode=True):
    if query_mode:
        sql = "SELECT * FROM " + table_name

        credentials = service_account.Credentials.from_service_account_file("./data_input/vti_sandbox.json")
        project_id = 'vti-sandbox'
        client = bigquery.Client(credentials=credentials, project=project_id)
        # Run a Standard SQL query with the project set explicitly
        df = client.query(sql, project=project_id).to_dataframe()
        df.to_csv("./data/input/bq_data_product")
    else:
        print("[INFO] Read input data from offline file")
        df = pd.read_csv('./data_input/data_product.csv')

    # information
    print_debug("[INFO] Table stock on hand: ", df.shape)
    print_debug("[INFO] Table stock on hand: ", df.head())
    return df


def Load_Data():
    # Load df product and store location
    df = Query_data_bq("soh", False)
    df_address = pd.read_csv('./data_input/aldo_location - aldo_store.csv')
    df = pd.merge(df, df_address[['store_name', 'address', 'latitude', 'longitude']], on='store_name')

    # change Aldo name into ez to read for user
    df[['store_index', 'store_name']] = df.store_name.str.split('-', expand=True)
    df.loc[~df.store_name.str.contains('Aldo'), 'store_name'] = 'Aldo' + df['store_name']
    df = df[~df.store_name.str.contains('Outlet')]

    # Split product name into like in fanpage
    df[['pro_name', 'code', 'season', 'something1', 'something2']] = df.product_name.str.split(' ', expand=True)
    df['pro_name'] = df['pro_name'].str.replace(" ", "")
    df['pro_name_color'] = df['pro_name'] + " " + df['color_code'].astype('str')

    pro_name = df.pro_name.unique().tolist()
    pro_name_color = df.pro_name_color.unique().tolist()
    color_code = df.color_code.unique().tolist()

    store = df[df.address.notnull()][['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name',
                                      'pro_name_color', 'full_price', 'current_price', 'color_code',
                                      'soh_ending_week', 'size']].drop_duplicates(subset='store_name').values.tolist()
    return df, pro_name, pro_name_color, color_code, store


# Don't decode if character in ignore words
ignore_words = ["?", "!", ":", "<", ">", "(", ")", "[", "]", "{", "}", "'", '"']

# Load data json ask and answer
with open('./data_input/data_as_an.json', 'r') as f:
    data_as_an = json.load(f)

# Load image suggestion
image_size_suggestion_path = 'https://i5.walmartimages.com/asr/b3de62b9-857f-4324-8ec2-0522c609e683_1.78db0aa71fe68c0f543313d43577c78e.png'

# load model LSTM classification
model = tf.keras.models.load_model('model_intent_classification_lstm_ver2.h5')

'''
    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------- SESSION 2 CREATE DATA PREPORCESSING  -----------------------------------------
'''
# Data Preprocessing to model
negative_emoticons = [':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{']
positive_emoticons = ['=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '😍', '✌', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)', ":)))))))", "😂"]


def word2charArr(words, pattern_words_dict):
    words = words.lower()
    arr = []

    pattern_words6 = pro_name  # intent find product code
    pattern_words8 = color_code  # intent find color code

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


from sklearn import preprocessing


def train_data_preprocessing():
    X_train, Y_train, pattern_words_dict = sentence2vec()
    X_train = pad_sequences(X_train, maxlen=100)
    highest_unicode = 8100

    cate_enc = preprocessing.LabelEncoder()
    LABEL = Y_train
    Y_train = cate_enc.fit_transform(Y_train)
    return X_train, Y_train, highest_unicode, LABEL, pattern_words_dict


'''
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------- SESSION 3 CREATE ACTION FOR CHATBOT ----------------------------------------
    # ------ 1. PREDICT INTENT/CONTEXT
    # ------ 2. FIND PRODUCT AVAILABLE
    # ------ 3. FIND NEAREST STORE ALDO
    # ------ 4. FIND PRODUCT PRICE
    # ------ 5. EXTRACT PRODUCT ID FROM IMAGE
'''


def get_random_message(model_return, answer_json):
    sample_responses = []
    quick_replies = [""]
    buttons = []
    for mess in answer_json['intents']:
        if mess['tag'] == model_return:
            sample_responses = mess['answer']
            quick_replies = mess['quick_replies']
            buttons = mess['buttons']
    res = random.choice(sample_responses).split("#")
    return res, quick_replies, buttons


# CORE OF CODE, Predict intent of sentence and reply
def model_predict_intent(sentence):
    train_data_preprocessing()
    data = []
    ERROR_THRESHOLD = 0.8
    data.append(word2charArr(sentence, pattern_words_dict))

    # Sentence ---> data input model
    inp = pad_sequences(np.array(data), maxlen=100)
    inp = np.where(inp <= highest_unicode, inp, 0)
    pred = model.predict(inp)

    # predict and get predict results
    prediction = tf.argmax(pred, 1)
    lookup = np.where(Y_train == prediction.numpy()[0])
    context = np.take(LABEL, lookup)[0][0]
    response, quick_replies, buttons = get_random_message(context, data_as_an)

    # sort by strength of probability
    results = pred[0]
    results = [[i, r] for i, r in enumerate(results)]
    results.sort(key=lambda x: x[1], reverse=True)
    if results[0][1] < ERROR_THRESHOLD:
        results[0][0] = -1
        response = ["Xin lỗi. Chatbot không hiểu ý bạn!"]
        quick_replies = [""]
        buttons = ["Cần hỗ trợ"]
    print_debug("DEBUG ANSWER MODE", results)
    return results[0][0], response, quick_replies, buttons


def ProductColorQuickReply(df_store):
    colors = df_store['color_code'].unique()
    quickReplies = list()
    for co in colors:
        if str(co) == "nan":
            pass
        else:
            quickReplies.append(co)
    return quickReplies


# Function to find product
def available_product_size_store(sentence):
    find_pro = []
    res = []
    quick_replies = [""]
    store_size_ava = []
    mode = 1
    for x1 in pro_name_color:
        if x1 in sentence.upper():
            find_pro.append(x1)
            mode = 1
    if find_pro == []:
        for x2 in pro_name:
            if x2 in sentence.upper():
                find_pro.append(x2)
                mode = 2

    find_pro = sorted(find_pro, key=len)

    if len(find_pro) != 0:
        find_pro = find_pro[-1]
    else:
        res.append(
            'Xin nhập lại mã sản phẩm! Có thể bạn nhập thiếu hoặc chưa nhập in hoa. Link sản phẩm: https://bitlylink.com/aldosanpham')
        quick_replies = ['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71', 'ADILASIEN 001']
        store_size_ava = []
        return res, quick_replies, store_size_ava

    if mode == 1:
        find_pro_color = find_pro
        find_pro = find_pro.split(' ')
        product_name, color_code = find_pro[0], find_pro[1]
        df_product = df[(df.product_name.str.contains(product_name)) &
                        (df.color_code.astype('str').str.contains(color_code))].sort_values('store_name')
        df_product = df_product[df_product.soh_ending_week > 0]
        if df_product.soh_ending_week.any() > 0:
            res.append('Sản phẩm {} còn ạ! Tìm size hoặc tìm cửa hàng.'.format(find_pro_color))
            quick_replies = ['Tìm size', 'Tìm cửa hàng', 'Giá sản phẩm']
            # Ouput store_size_ava available to lookup in next step
            df_store = df_product[['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name',
                                   'pro_name_color', 'full_price', 'current_price', 'color_code', 'soh_ending_week',
                                   'size']].drop_duplicates(subset=['store_name', 'product_id'])
            store_size_ava = df_store.values.tolist()
        else:
            res.append('Sản phẩm {} hiện đã hết rồi! Xin chọn một sản phẩm khác'.format(find_pro_color))
            quick_replies = ['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71', 'ADILASIEN 001']

    elif mode == 2:
        df_product = df[df.product_name.str.contains(find_pro)].sort_values('store_name')
        df_product = df_product[df_product.soh_ending_week > 0]
        if df_product.soh_ending_week.any() > 0:
            res.append("Sản phẩm {} còn ạ! Nhập mã màu bên cạnh VD: ASTIRASSA 95".format(find_pro))

            # Ouput store_list available to lookup in next step
            df_store = df_product[['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name',
                                   'pro_name_color', 'full_price', 'current_price', 'color_code', 'soh_ending_week',
                                   'size']].drop_duplicates(subset=['store_name', 'product_id'])
            store_size_ava = df_store.values.tolist()
            quick_replies = ProductColorQuickReply(df_product)

            if quick_replies == []:
                res = ['Sản phẩm {} còn ạ! Tìm size hoặc tìm cửa hàng.'.format(find_pro)]
                quick_replies = ['Tìm size', 'Tìm cửa hàng', 'Giá sản phẩm']
            print_debug("DEBUG quick replies avai", quick_replies)
        else:
            res.append('Sản phẩm {} hiện đã hết rồi! Xin chọn một sản phẩm khác'.format(find_pro))
            quick_replies = ['ASTIRASSA 95', 'SALARIA 301', 'DELUDITH 260', 'GELADA 973', 'AMUSA 961']
    # print(" -------------------- DEBUG available store size ava --------------------")
    print_debug("DEBUG available store size ava", res)
    return res, quick_replies, store_size_ava


# Function to find product color
def available_color(find_color, store_list):
    find_color = re.findall("[0-9]{1,4}", find_color)[0]
    store_array = np.array(store_list)
    res = []
    quick_replies = [""]
    store_size_ava = []
    if find_color not in color_code:
        res.append("Xin nhập lại mã màu! VD: 'ASTIRASSA 95'")
        store_size_ava = store_list

    else:
        df_product = pd.DataFrame(store_array,
                                  columns=['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name',
                                           'pro_name_color', 'full_price', 'current_price', 'color_code',
                                           'soh_ending_week', 'size'])
        df_product = df_product[df_product['color_code'].str.contains(find_color)]
        df_product = df_product[df_product['soh_ending_week'].astype("float32") > 0]
        if df_product['soh_ending_week'].astype("float32").any() > 0:
            res.append('Sản phẩm này còn ạ! Nhập "Tìm size" hoặc "Tìm cửa hàng"')
            # Ouput store_list available to lookup in next step
            df_store = df_product.drop_duplicates(subset=['store_name', 'product_id'])
            store_size_ava = df_store.values.tolist()
            quick_replies = ['Tìm size', 'Tìm cửa hàng']
        else:
            sen = 'Sản phẩm này chỉ có màu:'
            avai_colors = np.sort(np.unique(store_array[:, -3]))
            for c in avai_colors[0:-1]:
                sen = sen + ' ' + c + ','
            sen = sen + ' ' + avai_colors[-1]
            res.append(sen)
    return res, quick_replies, store_size_ava


def available_size(size_array):
    size_array = np.array(size_array)
    res = []
    store_list = []
    quick_replies = [""]
    pro_id = np.unique(size_array[:, 5]).tolist()

    if len(pro_id) != 1:
        res.append("Nhập mã sản phẩm ở trên bài viết để tìm Size. Link: https://bitlylink.com/aldosanpham'")
    elif any(x in size_array for x in ["Nosize", "No Size"]):
        res.append("Sản phẩm này không có size!")
        quick_replies = ["Quận 1, TP.HCM", "Quận 3, TP.HCM", "Quận 5, TP.HCM", "Quận 10, TP.HCM",
                         "Bình Thạnh, TP.HCM"]
    else:
        sen = 'Sản phẩm này còn size:'
        ava_size = np.sort(np.unique(size_array[:, -1]))
        for s in ava_size[0:-1]:
            sen = sen + ' ' + s + ','
        sen = sen + ' ' + ava_size[-1]
        res.append(sen)
        quick_replies = ["Quận 1, TP.HCM", "Quận 3, TP.HCM", "Quận 5, TP.HCM", "Quận 10, TP.HCM", "Bình Thạnh, TP.HCM"]
        store_list = size_array.tolist()
    return res, quick_replies, store_list


# Function find store
def find_coodinates(query):
    if len(query) < 13:
        scrape = ['0.0', '0.0']
    else:
        key = 'mykey'
        geocoder = OpenCageGeocode(key)
        results = geocoder.geocode(query)
        lat = results[0]['geometry']['lat']
        lng = results[0]['geometry']['lng']
        scrape = [lat, lng]
    return scrape


import re


def google_maps(address):
    if len(address) < 13:
        scrape = ['0.0', '0.0']
    else:
        req = address.replace(' ', '+')
        uri = 'https://www.google.com/maps/search/'
        url = uri + req
        ret = requests.get(url, timeout=10, allow_redirects=True).text
        try:
            scrape = ret.split('https://maps.google.com/maps/api/staticmap?center\u003d')[1].split('&amp;')[0].split(
                '%2C')
        except IndexError:
            try:
                scrape = find_coodinates(address)
            except IndexError:
                scrape = ['0.0', '0.0']
    return scrape


def find_neareast_store(address, store_list):
    cus = google_maps(address)
    nearest_store = []
    if cus == ['0.0', '0.0']:
        nearest_store.append('Không tìm thấy địa chỉ của bạn! Hãy thử nhập đầy đủ địa chỉ')
        return nearest_store
    else:
        lat1 = radians(float(cus[0]))
        lon1 = radians(float(cus[1]))
        results = []

        lowest_distance = 100
        # approximate radius of earth in km
        for i in store_list:
            R = 6373.0
            lat2 = radians(float(i[2]))
            lon2 = radians(float(i[3]))

            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))

            distance = R * c
            results.append(distance)
            if distance < lowest_distance:
                lowest_distance = distance
                nearest_store = i[0:2]
        results = sorted(np.array(results))
        if nearest_store == []:
            nearest_store.append(
                'Cửa hàng có sản phẩm hiện không có ở tỉnh/thành phố của bạn! Nhập lại địa chỉ 1 lần nữa để tìm cửa hàng gần bạn nhất nhưng không theo sản phẩm đã chọn.')
    return nearest_store


def find_product_id_price(sentence):
    find_pro = []
    mode = 1
    res = []
    quick_replies = [""]
    store_size_ava = []
    for x1 in pro_name_color:
        if x1 in sentence.upper():
            find_pro.append(x1)
            mode = 1
    if find_pro == []:
        for x2 in pro_name:
            if x2 in sentence.upper():
                find_pro.append(x2)
                mode = 2

    find_pro = sorted(find_pro, key=len)

    if len(find_pro) != 0:
        find_pro = find_pro[-1]
    else:
        res.append(
            'Xin nhập lại mã sản phẩm! Có thể bạn nhập thiếu hoặc chưa nhập in hoa. Link sản phẩm: https://bitlylink.com/aldosanpham')
        quick_replies = ['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71', 'ADILASIEN 001']
        store_size_ava = []
        return res, quick_replies, store_size_ava

    def respond_price():
        quick_replies = [""]
        store_size_ava = []
        if current_price < full_price:
            r = "Sản phẩm này có giá: {:,.0f} VND, nhưng đang được khuyến mãi {:.0%} chỉ còn {:,.0f} VND. Mua nhanh kẻo hết nào".format(
                full_price, discount, current_price)
            res.append(r)
            quick_replies = ['Tìm size', 'Tìm cửa hàng']
            # Ouput store_size_ava available to lookup in next step
            df_store = df_product[df_product.soh_ending_week > 0][
                ['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name',
                 'pro_name_color', 'full_price', 'current_price', 'color_code', 'soh_ending_week',
                 'size']].drop_duplicates(subset=['store_name', 'product_id'])
            store_size_ava = df_store.values.tolist()
        elif current_price == full_price:
            r = "Sản phẩm này có giá: {:,.0f} VND".format(full_price)
            res.append(r)
            quick_replies = ['Tìm size', 'Tìm cửa hàng']
            # Ouput store_size_ava available to lookup in next step
            df_store = df_product[df_product.soh_ending_week > 0][
                ['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name', 'pro_name_color',
                 'full_price', 'current_price', 'color_code', 'soh_ending_week',
                 'size']].drop_duplicates(subset=['store_name', 'product_id'])
            store_size_ava = df_store.values.tolist()
        elif (df_product.soh_ending_week.any() <= 0):
            res.append('Sản phẩm hiện đang hết hàng bạn vui lòng chọn 1 sản phẩm khác nha!')
        return res, quick_replies, store_size_ava

    if mode == 1:
        product_name, color_code = find_pro.split(" ")
        df_product = df[(df.product_name.str.contains(product_name)) & (
            df.color_code.astype('str').str.contains(color_code))].sort_values('store_name')
    else:
        df_product = df[df.product_name.str.contains(find_pro)].sort_values('store_name')

    # change type to float32
    df_product['soh_ending_week'] = df_product['soh_ending_week'].astype("float32")
    df_product['full_price'] = df_product['full_price'].astype("float32")
    df_product['current_price'] = df_product['current_price'].astype("float32")

    if df_product.soh_ending_week.any() > 0:
        full_price, current_price = df_product[['full_price', 'current_price']].iloc[0]
        discount = round(1 - (current_price / full_price), 2)
        full_price, current_price = int(full_price), int(current_price)
        res, quick_replies, store_size_ava = respond_price()
    else:
        res.append('Sản phẩm hiện đang hết hàng bạn vui lòng chọn 1 sản phẩm khác nha!')

    return res, quick_replies, store_size_ava


def find_price(price_array):
    find_pro = []
    res = []
    quick_replies = [""]
    store_size_ava = []
    price_array = np.array(price_array)

    # -------------------------------------------------- FUNCTION ------------------------------------------------------
    def respond_price():
        quick_replies = [""]
        store_size_ava = []
        if current_price < full_price:
            r = "Sản phẩm này có giá: {:,.0f} VND, nhưng đang được khuyến mãi {:.0%} chỉ còn {:,.0f} VND. Mua nhanh kẻo hết nào".format(
                full_price, discount, current_price)
            res.append(r)
            quick_replies = ['Tìm size', 'Tìm cửa hàng']
            # Ouput store_size_ava available to lookup in next step
            df_store = df_product[df_product.soh_ending_week > 0]
            df_store = df_store[['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name',
                                 'pro_name_color', 'full_price', 'current_price', 'color_code', 'soh_ending_week',
                                 'size']].drop_duplicates(subset=['store_name', 'product_id'])
            store_size_ava = df_store.values.tolist()
        elif current_price == full_price:
            r = "Sản phẩm này có giá: {:,.0f} VND".format(full_price)
            res.append(r)
            quick_replies = ['Tìm size', 'Tìm cửa hàng']
            # Ouput store_size_ava available to lookup in next step
            df_store = df_product[['store_name', 'address', 'latitude', 'longitude', 'product_id', 'pro_name',
                                   'pro_name_color', 'full_price', 'current_price', 'color_code', 'soh_ending_week',
                                   'size']].drop_duplicates(subset=['store_name', 'product_id'])
            store_size_ava = df_store.values.tolist()
        elif df_product.soh_ending_week.any() <= 0:
            res.append('Sản phẩm hiện đang hết hàng bạn vui lòng chọn 1 sản phẩm khác nha!')
        return res, quick_replies, store_size_ava

    # -------------------------------------------------- END ------------------------------------------------------
    if price_array.size == 0:
        res.append("Xin hãy nhập mã sản phẩm trên bài viết để tìm giá! Link sản phẩm https://bitlylink.com/aldosanpham")
        quick_replies = ['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71', 'ADILASIEN 001']
    else:
        product_id = np.unique(price_array[:, 5])
        if len(product_id) != 1:
            res.append(
                "Xin hãy nhập mã sản phẩm trên bài viết để tìm giá! Link sản phẩm https://bitlylink.com/aldosanpham")
            quick_replies = ['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71', 'ADILASIEN 001']
        else:
            df_product = pd.DataFrame(price_array,
                                      columns=['store_name', 'address', 'latitude', 'longitude', 'product_id',
                                               'pro_name', 'pro_name_color', 'full_price', 'current_price',
                                               'color_code', 'soh_ending_week', 'size'])
            # change type to float32
            df_product['soh_ending_week'] = df_product['soh_ending_week'].astype("float32")
            df_product['full_price'] = df_product['full_price'].astype("float32")
            df_product['current_price'] = df_product['current_price'].astype("float32")

            df_product = df_product[df_product['soh_ending_week'] > 0]
            if df_product['soh_ending_week'].any() > 0:
                full_price = df_product['full_price'].iloc[0]
                current_price = df_product['current_price'].iloc[0]
                discount = round(1 - (current_price / full_price), 2)
                full_price, current_price = int(full_price), int(current_price)
                res, quick_replies, store_size_ava = respond_price()
            else:
                res.append('Sản phẩm hiện đang hết hàng bạn vui lòng chọn 1 sản phẩm khác nha!')
                quick_replies = ['ERALESSA 650', 'SIERIAFLEX 001', 'KAIENIA 98', 'RPPL1B 680', 'COWIEN 71',
                                 'ADILASIEN 001']

    return res, quick_replies, store_size_ava


def get_Covid_19_Vietnam():
    response = []
    quick_replies = ["Thế giới"]
    url = "https://api.covid19api.com/summary"
    ret = requests.get(url, timeout=5)
    data = ret.json()["Countries"][-7]
    TotalConfirmed = "- Tổng số ca nhiễm đã xác nhận: {}".format(data['TotalConfirmed'])
    TotalDeaths = "- Tổng số người chết: {}".format(data['TotalDeaths'])
    NewRecovered = "- Số ca bình phục mới: {}".format(data['NewRecovered'])
    TotalRecovered = "- Tổng số ca bình phục: {}".format(data["TotalRecovered"])
    d1 = datetime.strptime(data['Date'], "%Y-%m-%dT%H:%M:%SZ")
    d1 = d1 + timedelta(hours=7)

    update_date = d1.strftime("%d/%m/%Y")
    update_hour = d1.strftime("%X")
    Update = "Cập nhật dữ liệu lúc {}, ngày {}".format(update_hour, update_date)

    source = "Nguồn: https://api.covid19api.com"

    res = "Cập nhật tình hình trong nước \n" + TotalConfirmed + "\n" + TotalDeaths + "\n" + NewRecovered + "\n" + TotalRecovered + "\n" + Update + "\n\n" + source
    response.append(res)
    return response, quick_replies


def get_Covid_19_global():
    response = []
    quick_replies = ["Việt Nam"]
    url = "https://api.covid19api.com/summary"
    ret = requests.get(url, timeout=5)
    data = ret.json()["Global"]
    TotalConfirmed = "- Tổng số ca nhiễm đã xác nhận: {:,.0f}".format(data['TotalConfirmed'])
    TotalDeaths = "- Tổng số người chết: {:,.0f}".format(data['TotalDeaths'])
    NewRecovered = "- Số ca bình phục mới: {:,.0f}".format(data['NewRecovered'])
    TotalRecovered = "- Tổng số ca bình phục: {:,.0f}".format(data["TotalRecovered"])

    source = "Nguồn: https://api.covid19api.com"
    res = "Cập nhật tình hình thế giới \n\n" + TotalConfirmed + "\n" + TotalDeaths + "\n" + NewRecovered + "\n" + TotalRecovered + "\n\n" + source
    response.append(res)
    return response, quick_replies


# Function image to text
def Image_to_product_code(image):
    image_to_text = pytesseract.image_to_string(image, lang='eng')

    sentence = image_to_text
    print_debug("Debug image text", sentence)

    if sentence == '':
        returned_code = bytesIOModule.BytesIO()

        image_file_Path = 'user_data/user_pic.jpg'
        searchUrl = 'http://www.google.hr/searchbyimage/upload'
        search_aldo = "&ei=xkakXsG7HIWC-Qbb-7WIBQ&q=aldo+shoes&oq=aldo+shoes"
        multipart = {'encoded_image': (image_file_Path, open(image_file_Path, 'rb')), 'image_content': ''}
        response = requests.post(searchUrl, files=multipart, allow_redirects=False)
        fetchUrl = response.headers['Location'] + search_aldo

        conn = pycurl.Curl()
        conn.setopt(conn.CAINFO, certifi.where())
        conn.setopt(conn.URL, str(fetchUrl))
        conn.setopt(conn.FOLLOWLOCATION, 1)
        conn.setopt(conn.USERAGENT,
                    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0')
        conn.setopt(conn.WRITEFUNCTION, returned_code.write)
        conn.perform()
        conn.close()
        code = returned_code.getvalue().decode('UTF-8')

        soup = BeautifulSoup(code, 'html.parser')

        search_tittles = []
        find_pro = []
        for div in soup.findAll('div', attrs={'class': 'rc'}):
            search_tittles.append(str(div.find('h3')))
        print_debug("H3", search_tittles)
        if search_tittles:
            for search_tittle in search_tittles:
                for p in pro_name:
                    if p in search_tittle.upper():
                        find_pro.append(p)

        if find_pro:
            sentence = find_pro[0]
        else:
            sentence = ""
    return sentence

def recent_promotion():
    get_promotion = collection3.find_one(sort=[('_id', pymongo.DESCENDING)])
    quick_replies = ["Tìm sản phẩm"]
    if get_promotion['recent_promotion']:
        return [get_promotion['recent_promotion']], quick_replies
    else:
        return ["Hiện tại không có chương trình khuyến mãi nào ạ!"], quick_replies



def Auto_Marketing_Started(sentence):
    extract_sentence = sentence.split("--")
    am, access_token_auto_mark, mode_auto_mark = extract_sentence[0], extract_sentence[1], extract_sentence[2]
    get_am_data = collection4.find_one(sort=[('_id', pymongo.DESCENDING)])
    auto_mark_post_back = []
    get_am_data = get_am_data['started']
    content, quick_replies, elements = get_am_data["content"], get_am_data["quick_replies"], get_am_data["elements"]

    def send_Auto_Mark_to_list(recipient_id_list):
        """typing it on messenger to action
        Syntax: auto_marketing--password--MODE"""
        if quick_replies != [""]:
            for psid in recipient_id_list:
                print(psid, content, quick_replies)
                send_Message_with_QuickReply(psid, content, quick_replies)
                time.sleep(TIME_SLEEP)
        elif "buttons" in elements[0]:
            for psid in recipient_id_list:
                print(psid, elements)
                sendCarouselMsg(psid, elements)
                time.sleep(TIME_SLEEP)
        else:
            return "There is no auto marketing"

    if access_token_auto_mark == "auto_marketing123456":
        print_debug("Password valid", access_token_auto_mark)

        # Mode Debug
        if mode_auto_mark == "Debug":
            get_admin = collection2.find_one(sort=[('_id', pymongo.ASCENDING)])
            print_debug("get_admin", get_admin)
            send_Auto_Mark_to_list([get_admin['id']])
        # Mode Test internal
        elif mode_auto_mark == "Test":
            recipient_id_Test = ["3079512872106958"]
            send_Auto_Mark_to_list(recipient_id_Test)

        # Mode Send All
        elif mode_auto_mark == "All_Mess":
            recipient_id_All = [x['id'] for x in collection2.find({}, {"_id": 0, "first_name": 0, "last_name": 0, "gender": 0})]
            send_Auto_Mark_to_list(recipient_id_All)
            auto_mark_post_back.append(content)

        # Mode Send All Interact
        elif mode_auto_mark == "All_Page_Interact":
            recipient_id_All = [x['id'] for x in collection2.find({}, {"_id": 0, "first_name": 0, "last_name": 0, "gender": 0})]
            send_Auto_Mark_to_list(recipient_id_All)
            auto_mark_post_back.append(content)

    else:
        return "Invalid password Auto Mark"

'''
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------- FUNCTION TO SEND MESSAGE -------------------------------------------------
'''


# Person profile information
def getPersonInfo(psid):
    """Get user info by facebook doc
    link: https://developers.facebook.com/docs/graph-api/reference/user/?locale=vi_VN """
    url = "https://graph.facebook.com/v4.0/%s" % psid
    payload = {"fields": "first_name,last_name,gender,email,address,age_range", "access_token": PAGE_ACCESS_TOKEN}
    r = requests.get(url, params=payload)
    return r.json()


def sendTypingBubble(psid):
    url = "https://graph.facebook.com/v4,0/me/messages"
    requestBody = {"recipient": {"id": psid},
                   "sender_action": "typing_on"}
    r = requests.post(url, params={"access_token": PAGE_ACCESS_TOKEN}, json=requestBody)
    # time.sleep(4.0)


# uses PyMessenger to send response to user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"


def sendQuickReply(psid, msgText, quickReplies):
    url = "https://graph.facebook.com/v4.0/me/messages"
    requestBody = {"recipient": {"id": psid},
                   "message": {"text": msgText,
                               "quick_replies": quickReplies
                               }
                   }
    r = requests.post(url, params={"access_token": PAGE_ACCESS_TOKEN}, json=requestBody)
    # print(r.url)


def send_Message_with_QuickReply(psid, msgText, quick_replies):
    if quick_replies == [""]:
        send_message(psid, msgText)
    else:
        quickReplies = []
        for quick_reply in quick_replies:
            quickReplies.append({"content_type": "text", "title": quick_reply, "payload": "quick_replies"})
        print_debug("send MESSAGE with QUICKREPLY", msgText)
        print_debug("send MESSAGE with QUICKREPLY", quickReplies)
        sendQuickReply(psid, msgText, quickReplies)


# Generic button to send text
def send_Message_with_Buttons(psid, text, title_buttons):
    buttons = []
    for title in title_buttons:
        buttons.append({"type": "postback", "title": title, "payload": "send_message_option"})
    bot.send_button_message(psid, text, buttons)


# Buttons for answer mode 12
def send_google_map_Buttons(psid, store_address):
    buttons = []
    req = store_address.replace(' ', '+')
    uri = 'https://www.google.com/maps/search/'
    url = uri + req
    buttons.append({"type": "web_url", "url": url, "title": "Mở google map"})
    bot.send_button_message(psid, store_address, buttons)


'''
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------- ADD QUICK REPLY AND CAROUSEL -----------------------------------------------
'''


# Generic Carousel message
def sendCarouselMsg(psid, elements):
    url = "https://graph.facebook.com/v4.0/me/messages"
    requestBody = {"recipient": {"id": psid},
                   "message": {"attachment": {"type": "template",
                                              "payload": {"template_type": "generic",
                                                          "elements": elements}
                                              }
                               }

                   }

    r = requests.post(url, params={"access_token": PAGE_ACCESS_TOKEN}, json=requestBody)


# List message
def sendListMsg(psid, elements):
    url = "https://graph.facebook.com/v4.0/me/messages"
    requestBody = {"recipient": {"id": psid},
                   "message": {"attachment": {"type": "template",
                                              "payload": {"template_type": "list",
                                                          "top_element_style": "compact",
                                                          "elements": elements}
                                              }
                               }

                   }

    r = requests.post(url, params={"access_token": PAGE_ACCESS_TOKEN}, json=requestBody)


# Quick replies according to the product category
def sendCategoriesQuickReply(psid, text):
    quickReplies = [{"content_type": "text", "title": "Giày nữ", "payload": "shoes_women"},
                    {"content_type": "text", "title": "Túi xách, ví nữ", "payload": "bags"},
                    {"content_type": "text", "title": "Giày nam", "payload": "shoes_men"}]
    sendQuickReply(psid, text, quickReplies)


''' # -------------------------------- MAIN MODEL - ADD OR REMOVE FUNCTION HERE ------------------------------------ '''

mode_send_shoes = ['giày nữ']
mode_send_bags = ['túi xách']
mode_send_accessories = ['giày nam']

# Patten_words 4 is product CODE
mode_send_product_code = ['mã', 'code', 'mẫu']
mode_send_category = ['danh mục']
mode_find_product = ['sản phẩm', 'tra cứu', 'còn']


# Keywords_address = ['']


# Handle messages events
def handleMessage(sender_pdid, message, user_session):
    """
    input: type(json) user session for processing
    {"store_list":List(store info, product),'human_mode':List(False, Last_message, current time)}
    output: session type(json)
    """
    store_list = user_session['store_list']
    human_mode = user_session['human_mode']
    user_name = user_session['user_name']
    auto_mark = user_session['auto_marketing']
    if 'postback' in message:
        received_message = message['postback']
    elif 'message' in message:
        received_message = message['message']
    else:
        return user_session

    # skip echo messages
    if 'is_echo' in received_message:
        print_debug("ECHO == TRUE", received_message.get('text'))
        current_time = datetime.now().timestamp()
        if (received_message.get('text') != human_mode[1]) or ((current_time - human_mode[2]) / 60 > 2):
            human_mode[0] = False
        if auto_mark[0]:
            auto_mark[1] = "have_read"
        return user_session


    elif human_mode[0]:
        print("TALK TO HUMAN", "-----------------------")
        return user_session
    else:
        # IF not ECHO message, and not Human Mode
        if 'text' in received_message:
            sentence = received_message['text']
            print_debug("Debug sentence text", sentence)
        elif 'title' in received_message:
            sentence = received_message['title']
            print_debug("Debug sentence title", sentence)
        elif 'attachments' in received_message:
            print_debug("DEBUG ATTACHMENTS", received_message.get('attachments'))
            if 'image' in received_message.get('attachments')[0]['type']:
                image_url = received_message['attachments'][0]['payload']['url']
                # download image
                with open('./user_data/user_pic.jpg', 'wb') as user_image:
                    response = requests.get(image_url, stream=True, timeout=5)
                    if not response.ok:
                        print(response)

                    for block in response.iter_content(1024):
                        if not block:
                            print_debug("DEBUG IMAGE", "Image is not available")
                        else:
                            user_image.write(block)

                # open image & extract to product id
                image = Image.open('./user_data/user_pic.jpg')
                sentence = Image_to_product_code(image)
        else:
            sentence = "Nói chuyện với người"

        tokenizeWords = word_tokenize(sentence.lower())
        qr = []
        # Mode AUTO MARKETING STARTED CAMPAIGN
        if re.search('auto_marketing+--+.+--+.', sentence):
            Auto_Marketing_Started(sentence)
            auto_mark[0] = True
            qr.append("1")


        # Mode find PRODUCT_CODE
        elif any(x.upper() in pro_name for x in tokenizeWords) & ("giá" not in sentence.lower()):
            s_response, quick_replies, store_product = available_product_size_store(sentence)
            if s_response != []:
                send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)
                qr.append(1)
            if store_product != []:
                store_list = store_product

        # Mode find PRODUCT_PRICE
        elif any(x.upper() in pro_name for x in tokenizeWords) & ("giá" in sentence.lower()):
            s_response, quick_replies, store_product = find_product_id_price(sentence)
            if s_response != []:
                send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)
                qr.append(1)
            if store_product != []:
                store_list = store_product

        # Mode send SHOES WOMEN
        elif "giày nữ" in sentence.lower() and (len(tokenizeWords) < 6):
            send_message(sender_pdid, "Chọn một sản phẩm dưới đây")
            sendCarouselMsg(sender_pdid, shoes_women_elements)
            qr.append("1")

        # Mode send BAGS
        elif any(x in sentence.lower() for x in ["túi xách", "ví nữ"]) and (len(tokenizeWords) < 6):
            send_message(sender_pdid, "Chọn một sản phẩm dưới đây")
            sendCarouselMsg(sender_pdid, bags_elements)
            qr.append("1")

        # Mode send SHOES MEN
        elif "giày nam" in sentence.lower() and (len(tokenizeWords) < 6):
            send_message(sender_pdid, "Chọn một sản phẩm dưới đây")
            sendCarouselMsg(sender_pdid, shoes_men_elements)
            qr.append("1")

        # Mode send CATEGORIES
        elif (any(x in mode_send_category for x in tokenizeWords) == True) and (len(tokenizeWords) < 8):
            sendCategoriesQuickReply(sender_pdid, "Chọn một danh mục dưới đây: ")
            qr.append("1")

        # IF NOT ANY MODE ON USING MODEL
        if qr == []:
            answer_mode, f_response, quick_replies, title_buttons = model_predict_intent(sentence)

            if title_buttons != []:
                if len(f_response) == 1:
                    send_Message_with_Buttons(sender_pdid, f_response[0], title_buttons)
                elif len(f_response) == 2:
                    send_message(sender_pdid, f_response[0])
                    send_Message_with_Buttons(sender_pdid, f_response[1], title_buttons)
            elif quick_replies != [""]:
                if len(f_response) == 1:
                    send_Message_with_QuickReply(sender_pdid, f_response[0].format(user_name), quick_replies)
                elif len(f_response) == 2:
                    send_message(sender_pdid, f_response[0])
                    send_Message_with_QuickReply(sender_pdid, f_response[1], quick_replies)
            else:
                if len(f_response) == 1:
                    send_message(sender_pdid, f_response[0])
                elif len(f_response) == 2:
                    send_message(sender_pdid, f_response[1])

            # Mode find product_id
            if (answer_mode == 7) | (answer_mode == 8):
                s_response, quick_replies, store_product = available_product_size_store(sentence)
                store_list = store_product
                if s_response != []:
                    send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)

            elif (answer_mode == 9) | (answer_mode == 10):  # Mode find COLORS
                s_response, quick_replies, store_product = available_color(sentence, store_list)
                store_list = store_product
                if s_response != []:
                    send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)

            # Mode find SIZE
            elif answer_mode == 11:
                s_response, quick_replies, store_product = available_size(store_list)
                store_list = store_product
                if s_response != []:
                    send_message(sender_pdid, s_response[0])
                    send_Message_with_QuickReply(sender_pdid, "Nhập địa chỉ để tìm cửa hàng gần nhất", quick_replies)

            # Mode store search address
            elif answer_mode == 13:  # any(x in keyword_address for x in tokenizeWords): # Mode find STORE
                s_response = find_neareast_store(sentence, store_list)
                store_list = store
                if s_response != []:
                    send_message(sender_pdid, s_response[0])
                if len(s_response) == 2:
                    send_google_map_Buttons(sender_pdid, s_response[1])

            # Mode Suggest SIZE send image
            elif answer_mode == 14:
                bot.send_image_url(sender_pdid, image_size_suggestion_path)

            # Mode Turn True human_mode
            elif answer_mode == 15:
                current_time = datetime.now().timestamp()
                human_mode = [True, f_response[0], current_time]

            # Mode product_id price
            elif answer_mode == 29:
                s_response, quick_replies, store_product = find_product_id_price(sentence)
                store_list = store_product
                if s_response != []:
                    send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)

            # Mode find price
            elif answer_mode == 30:
                s_response, quick_replies, store_product = find_price(store_list)
                store_list = store_product
                if s_response != []:
                    send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)

            # Mode get Covid-19 Việt Nam
            elif answer_mode == 32:
                s_response, quick_replies = get_Covid_19_Vietnam()
                if s_response != []:
                    send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)

            # Mode get Covid-19 World
            elif answer_mode == 33:
                s_response, quick_replies = get_Covid_19_global()
                if s_response != []:
                    send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)

            # Mode get recency promotion available on facebook fanpage
            elif answer_mode == 34:
                s_response, quick_replies = recent_promotion()
                if s_response != []:
                    send_Message_with_QuickReply(sender_pdid, s_response[0], quick_replies)
        # ouput --> session
        if store_list != []:
            user_session['store_list'] = store_list
        user_session['human_mode'] = human_mode
        return user_session


'''
    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- SESSION 4 VERIFY TOKEN -------------------------------------------------
'''


def verify_fb_token(token_sent):
    # take token sent by facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


'''
    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- SESSION 5 MAIN APP -------------------------------------------------
'''

# DEFINE
app = Flask(__name__)
app.secret_key = 'supersecretkey'

PAGE_ACCESS_TOKEN = 'PAGE ACCESS TOKEN'
VERIFY_TOKEN = 'supersecretkey'

bot = Bot(PAGE_ACCESS_TOKEN)

# MongoDB
cluster = MongoClient("mongodb+srv://user:password@cluster0-0eopm.mongodb.net/test?retryWrites=true&w=majority")
db = cluster.get_database('messenger_chatbot')
collection1 = db.user_message
collection2 = db.user_info
collection3 = db.recent_promotion
collection4 = db.auto_marketing

# Using tesseract to parse image
pytesseract.pytesseract.tesseract_cmd = r"./data_input/Tesseract-OCR/tesseract.exe"

# Query mode
TIME_QUERY = 1.0
TIME_SLEEP = 1
BATCH = 569

# We will receive messages that Facebook sends our bot at this endpoint
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook."""
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    # if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if ("message" in message) | ("postback" in message):
                    # Insert new "message" to Mongo DB (Atlas)
                    collection1.insert_one(message)
                    print_debug("MESSAGE", message)

                    # Get Facebook ID and User Info
                    recipient_id = message['sender']['id']
                    user_id = str(recipient_id)
                    print_debug("DEBUG message", message)

                    user_info = getPersonInfo(recipient_id)
                    print_debug("DEBUG get user info", user_info)

                    if 'error' not in user_info:  # if not error get user info
                        f_name = user_info.get('first_name')
                        l_name = user_info.get('last_name')
                        user_name = str(l_name) + " " + str(f_name)
                    else:
                        user_name = ""

                    '''----------------------------------------------'''
                    """In BUTTON POSTBACK sender ID will be chatbot but not user
                        So switch recipient_id"""
                    if recipient_id == "775081399353381":
                        user_id = str(message['recipient']['id'])

                    ''' # ==========================================================================================='''
                    ''' # -------------------------------------------------------------------------------------------'''
                    ''' # -------------------------------------- GET ANSWER -----------------------------------------'''
                    current_time = round(datetime.now().timestamp())

                    global df, pro_name, pro_name_color, color_code, store, TIME_QUERY
                    global X_train, Y_train, highest_unicode, LABEL, pattern_words_dict
                    global BATCH
                    ''' # ----------------------------- SETTING QUERY NEW DATA ----------------------------------'''
                    DoW = 1
                    DAYS = 7
                    if (date.fromtimestamp(current_time).weekday() == DoW) and (current_time - TIME_QUERY > 86500):
                        df, pro_name, pro_name_color, color_code, store = Load_Data()
                        X_train, Y_train, highest_unicode, LABEL, pattern_words_dict = train_data_preprocessing()
                        TIME_QUERY = current_time
                    elif current_time - TIME_QUERY > 86400 * DAYS:
                        df, pro_name, pro_name_color, color_code, store = Load_Data()
                        X_train, Y_train, highest_unicode, LABEL, pattern_words_dict = train_data_preprocessing()
                        TIME_QUERY = current_time
                    ''' # ---------------------------------------------------------------------------------------'''

                    # Load user session: store, product code, color code, soh, size, human_mode
                    try:
                        with open('./user_data/user_session.json', 'r') as session_in_file:
                            session = json.load(session_in_file)
                        try:
                            user_session_in = session[user_id]
                        except KeyError:
                            new_user = {user_id: {"store_list": store, "human_mode": [False, "f_mess", current_time],
                                                  "user_name": user_name, "auto_marketing": [False, "no"]}}
                            session.update(new_user)
                            user_session_in = session[user_id]

                            # if new user send user info to mongoDB
                            collection2.insert_one(user_info)

                            # # Delete Cache session if two many user
                            # if len(session) > 200:
                            #     session_sort = sorted(session.item(), key=lambda kv: kv[1]["human_mode"][2])
                            #     del_cache = session_sort[-1][0]
                            #     session.pop(del_cache)
                    except FileNotFoundError:
                        session = {user_id: {"store_list": store, "human_mode": [False, "f_mess", current_time],
                                             "user_name": user_name, "auto_marketing": [False, "no"]}}
                        user_session_in = session[user_id]

                        # if new user send user info to mongoDB
                        collection2.insert_one(user_info)
                    except ValueError:
                        continue

                    ''' # --------------------------------- FUNCTION HANDLE MESSAGE ---------------------------------'''
                    ''' #                                                                                            '''
                    user_session_out = handleMessage(recipient_id, message, user_session_in)

                    ''' # -------------------------------------------------------------------------------------------'''
                    session[user_id] = user_session_out
                    with open('./user_data/user_session.json', 'w') as session_out_file:
                        json.dump(session, session_out_file, indent=4)

                    ''' # ---------------------------------- END OF GET ANSWER --------------------------------------'''
                    ''' # -------------------------------------------------------------------------------------------'''
                    ''' # ==========================================================================================='''
            return "Message Processed"


DEBUG = False

def print_debug(Msg: str, variable):
    debug_mode = DEBUG
    if debug_mode:
        print(" -------------------- {} --------------------".format(Msg))
        print(variable)
        print(' ')
        print(' ')


if __name__ == "__main__":
    app.run(debug=DEBUG)
## https://developers.facebook.com/apps/507744133452111/messenger/settings/
# How to deploy flask app on HOROKU link:
# https://www.youtube.com/watch?v=eH1DAQlDuYw
# https://www.youtube.com/watch?v=sqJSdJbOOU0
# C:\Users\Misle\Anaconda3\Lib\site-packages
# ngrok http -config=ngrok.yml 5000
# run: cd C:\Users\Misle\host

## run code on heroku locally
# Start a console
# heroku run python messenger_chatbot_aldo.py shell
# Todo: Add auto marketing send in messenger
# Todo: campaign started
# Todo: send after
# Todo: read or not read
# Todo: engagement impression or clicked
# Todo: Test read
# Todo Test click link