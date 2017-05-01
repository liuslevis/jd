#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os
import math
import numpy as np

action_paths = './data/raw/JData_Action_%s.csv' 
comment_path = './data/raw/JData_Comment.csv'
product_path = './data/raw/JData_Product.csv'
user_path    = './data/raw/JData_User.csv'

ACTION_TYPES = 6

comment_date = ['20160201', '20160208', '20160215', '20160222', '20160229', '20160307', '20160314', '20160321', '20160328', '20160404', '20160411', '20160415']

# d1 ~ d2 训练数据 d3 ~ d4标签
d1 = '20160201'
d2 = '20160214'
d3 = '20160215'
d4 = '20160219'


def strptime(dt_str):
    return datetime.strptime(dt_str.replace('-', ''), '%Y%m%d')

def convert_age(age_str):
    if age_str == '-1':
        return 0
    elif age_str == '15岁以下':
        return 1
    elif age_str == '16-25岁':
        return 2
    elif age_str == '26-35岁':
        return 3
    elif age_str == '36-45岁':
        return 4
    elif age_str == '46-55岁':
        return 5
    elif age_str == '56岁以上':
        return 6
    else:
        return -1

def convert_reg_tm(reg_tm):
    if reg_tm < -1:
        return 0
    elif reg_tm <= 30 * 1:
        return 1
    elif reg_tm <= 30 * 6:
        return 2
    elif reg_tm <= 30 * 12:
        return 3
    elif reg_tm <= 30 * 24:
        return 4
    else:
        return 5

# def convert_reg_tm(reg_tm):

def get_user(d1, d2, d3, d4):
    cache_path = './cache/user_%s_%s_%s_%s.pkl' % (d1, d2, d3, d4)
    if os.path.exists(cache_path):
        df = pickle.load(open(cache_path, 'rb'))
        return df
    else:
        df = pd.read_csv(user_path)
        df['age'] = df['age'].map(convert_age)
        df['user_reg_tm'] = df['user_reg_tm']\
            .map(lambda reg_tm : (strptime(d4) - strptime(reg_tm)).days if type(reg_tm) is str else -1)\
            .map(convert_reg_tm)
        # feats = [pd.get_dummies(df[col], prefix=col) for col in ['age', 'sex', 'user_lv_cd', 'user_reg_tm']]
        # df = pd.concat([df['user_id'], feats[0], feats[1], feats[2], feats[3]], axis=1)
        pickle.dump(df, open(cache_path, 'wb'))
        return df

def get_product():
    cache_path = './cache/product.pkl'
    if os.path.exists(cache_path):
        df = pickle.load(open(cache_path, 'rb'))
        return df
    else:
        df = pd.read_csv(product_path)
        # feats = [pd.get_dummies(df[col], prefix=col) for col in ['a1', 'a2', 'a3', 'cate']] # brand
        # df = pd.concat([df['sku_id', 'brand'], feats[0], feats[1], feats[2], feats[3]], axis=1)
        pickle.dump(df, open(cache_path, 'wb'))
        return df

def inv_dict(d):
    return dict((v,k) for k,v in d.items())

def parse_action_line(line):
    parts = line.rstrip().split(',')
    if len(parts) < 7:
        print('invalid line:', line)
    user_id, sku_id, time, model_id, action, cate, brand = parts
    user_id = int(float(user_id))
    sku_id = int(sku_id)
    model = int(model_id) if len(model_id) > 0 else -1
    action = int(action)
    cate = int(cate) if len(cate) > 0 else -1
    brand = int(brand) if len(brand) > 0 else -1
    return user_id, sku_id, time, model_id, action, cate, brand

user = get_user(d1, d2, d3, d4)
product = get_product()

index_user = user['user_id'].to_dict() # index:user_id
index_product = product['sku_id'].to_dict() #index:product_id
user_index    = inv_dict(index_user)   # user_id:index
product_index = inv_dict(index_product) # sku_id :index


user_item_train = {} # {i:j}
user_item_label = np.zeros((len(user), len(product))) # M[i=user][j=item] = label
user_item_action_ = [np.zeros((len(user), len(product))) for i in range(1 + ACTION_TYPES)] # M[type][i=user][j=item] = sum

dates = list(set(map(lambda d:d[:-2], [d1, d2, d3, d4])))

with open(action_paths % dates[0]) as f:
    for line in f.readlines():
        if line.startswith('user_id,sku_id,time,model_id,type,cate,brand'):
            continue
        user_id, sku_id, time, model_id, type_, cate, brand = parse_action_line(line)
        date = time.split(' ')[0].replace('-', '')
        
        if d1 <= date <= d4 and sku_id in product_index:
            i = user_index[user_id]
            j = product_index[sku_id]
            if 1 <= type_ <= 6 and d1 <= date <= d2:
                user_item_action_[type_][i][j] += 1
                user_item_train.update({i:j})

            if type_ == 4 and d3 <= date <= d4: # buy
                user_item_label[i][j] = 1

label = []
act_1 = []
act_2 = []
act_3 = []
act_4 = []
act_5 = []
act_6 = []
sku_ids = []
user_ids = []
user_age = []
user_sex = []
user_lv_cd = []
user_reg_tm = []
sku_a1 = []
sku_a2 = []
sku_a3 = []
sku_cate = []
sku_brand = []

for i, j in user_item_train.items():

    label.append(np.int32(user_item_label[i][j]))
    act_1.append(np.int32(user_item_action_[1][i][j]))
    act_2.append(np.int32(user_item_action_[2][i][j]))
    act_3.append(np.int32(user_item_action_[3][i][j]))
    act_4.append(np.int32(user_item_action_[4][i][j]))
    act_5.append(np.int32(user_item_action_[5][i][j]))
    act_6.append(np.int32(user_item_action_[6][i][j]))

    user_id = np.int32(index_user[i])
    user_row = user.iloc[[i]]
    sku_id = np.int32(index_product[j])
    sku_row = product.iloc[[j]]

    user_ids.append(np.int32(index_user[i]))
    user_sex.append(np.int32(user_row['sex'].values[0]))
    user_age.append(np.int32(user_row['age'].values[0]))
    user_lv_cd.append(np.int32(user_row['user_lv_cd'].values[0]))
    user_reg_tm.append(np.int32(user_row['user_reg_tm'].values[0]))

    sku_ids.append(np.int32(sku_id))
    sku_a1.append(np.int32(sku_row['a1'].values[0]))
    sku_a2.append(np.int32(sku_row['a2'].values[0]))
    sku_a3.append(np.int32(sku_row['a3'].values[0]))
    sku_cate.append(np.int32(sku_row['cate'].values[0]))
    sku_brand.append(np.int32(sku_row['brand'].values[0]))

df = pd.DataFrame({
    'label':label, 
    'user_id':user_ids, 
    'sku_id':sku_ids, 
    'act_1':act_1,
    'act_2':act_2,
    'act_3':act_3,
    'act_4':act_4,
    'act_5':act_5,
    'act_6':act_6,
    'user_sex':user_sex,
    'user_age':user_age,
    'user_lv_cd':user_lv_cd,
    'user_reg_tm':user_reg_tm,
    'sku_a1':sku_a1,
    'sku_a2':sku_a2,
    'sku_a3':sku_a3,
    'sku_cate':sku_cate,
    'sku_brand':sku_brand,
    })

feats = [pd.get_dummies(df[col], prefix=col) for col in ['user_sex', 'user_age', 'user_lv_cd', 'user_reg_tm', 'sku_a1', 'sku_a2', 'sku_a3']]
df = pd.concat([df[['label', 'act_1', 'act_2', 'act_3', 'act_4', 'act_5', 'act_6']], feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6]], axis=1)
  
df.to_csv('data/input/train_%s_%s_%s_%s.csv' % (d1, d2, d3, d4), index=False)

# def make_train_data(d1, d2, d3, d4):
# pass
# train_data = make_train_data(d1, d2, d3, d4)

