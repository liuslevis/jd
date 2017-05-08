#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
from datetime import timedelta
import pandas as pd
import pickle
import os,sys
import math
import numpy as np

action_paths = './data/raw/JData_Action_%s.csv' 
comment_path = './data/raw/JData_Comment.csv'
product_path = './data/raw/JData_Product.csv'
user_path    = './data/raw/JData_User.csv'
train_path   = 'data/input/train_%s_%s_%s_%s.csv'

USE_CACHE = True
ACTION_TYPES = 6

comment_date = ['20160201', '20160208', '20160215', '20160222', '20160229', '20160307', '20160314', '20160321', '20160328', '20160404', '20160411', '20160415']

def ndays_after(ndays, date_str):
    return datetime.strftime(datetime.strptime(date_str, '%Y%m%d') + timedelta(days=ndays), '%Y%m%d')

def strptime(dt_str):
    return datetime.strptime(dt_str.replace('-', ''), '%Y%m%d')

def convert_age(age_str):
    if age_str == '-1':
        return -1
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

def get_user_df(d1, d2, d3, d4):
    cache_path = './cache/user_%s.pkl' % (d4)
    if os.path.exists(cache_path) and USE_CACHE:
        return pickle.load(open(cache_path, 'rb'))
    else:
        df = pd.read_csv(user_path)
        df['age'] = df['age'].map(convert_age)
        df['sex'] = df['sex'].fillna(-1)
        df['user_reg_tm'] = df['user_reg_tm']\
            .map(lambda reg_tm : (strptime(d4) - strptime(reg_tm)).days if type(reg_tm) is str else -1)\
            .map(convert_reg_tm)
        pickle.dump(df, open(cache_path, 'wb'))
        return df

def get_user(d1, d2, d3, d4):
    df = get_user_df(d1, d2, d3, d4)
    dict1 = {}
    dict2 = {}
    for index, row in df.iterrows():
        user_id = df.ix[index, 'user_id']
        info = {col:df.ix[index, col] for col in df.columns}
        info.update({'index':index})
        dict1.update({user_id:info})
        dict2.update({index:info})
    ret = {}
    ret.update({'user_id':dict1})
    ret.update({'index':dict2})
    return ret

def get_product_df():
    cache_path = './cache/product.pkl'
    if os.path.exists(cache_path) and USE_CACHE:
        return pickle.load(open(cache_path, 'rb'))
    else:
        df = pd.read_csv(product_path)
        pickle.dump(df, open(cache_path, 'wb'))
        return df

#{'sku_id':{sku_id:info}, 'index':{index:info}}
def get_product():
    df = get_product_df()
    dict1 = {}
    dict2 = {}
    for index, row in df.iterrows():
        sku_id = df.ix[index,'sku_id']
        info = {col:df.ix[index, col] for col in df.columns}
        info.update({'index':index})
        dict1.update({sku_id:info})
        dict2.update({index:info})
    ret = {}
    ret.update({'sku_id':dict1})
    ret.update({'index':dict2})
    return ret

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

# d1 ~ d2 训练数据 d3 ~ d4标签
def make_train_data(d1, d2, d3, d4):
    user = get_user(d1, d2, d3, d4)
    product = get_product()
    user_len = len(user['user_id'])
    product_len = len(product['sku_id'])

    user_item_train = {} # {i:j}
    user_item_label = np.zeros((user_len, product_len)) # M[i=user][j=item] = label
    user_item_action_ = np.zeros((ACTION_TYPES+1, user_len, product_len)) # M[type][i=user][j=item] = sum
    user_a_ = np.zeros((4, user_len, product_len))

    dates = list(set(map(lambda d:d[:-2], [d1, d2, d3, d4])))
    for date in dates:
        with open(action_paths % date) as f:
            for line in f.readlines():
                if line.startswith('user_id,sku_id,time,model_id,type,cate,brand'):
                    continue
                user_id, sku_id, time, model_id, type_, cate, brand = parse_action_line(line)
                date = time.split(' ')[0].replace('-', '')
                
                if d1 <= date <= d4 and sku_id in product['sku_id']:
                    i = user['user_id'][user_id]['index']
                    j = product['sku_id'][sku_id]['index']

                    if 1 <= type_ <= 6 and d1 <= date <= d2:
                        if type_ >= user_item_action_.shape[0] or i >= user_item_action_.shape[1] or j >= user_item_action_.shape[2]:
                            print('debug', user_item_action_.shape, type_, i, j)
                        user_item_action_[type_][i][j] += 1
                        user_item_train.update({i:j})

                    if type_ == 4 and d3 <= date <= d4: # buy
                        user_item_label[i][j] = 1
                        user_item_train.update({i:j})

                    user_a_[1][i][j] = product['index'][j]['a1']
                    user_a_[2][i][j] = product['index'][j]['a2']
                    user_a_[3][i][j] = product['index'][j]['a3']
                    
    columns = [
        'label',
        'user_id',
        'sku_id',
        'act_1',
        'act_2',
        'act_3',
        'act_4',
        'act_5',
        'act_6',

        'user_sex',
        'user_age',
        'user_lv_cd',
        'user_reg_tm',

        'sku_a1',
        'sku_a2',
        'sku_a3',
        'sku_cate',
        'sku_brand',

        # 'user_a1',
        # 'user_a2',
        # 'user_a3',
        ]

    table = []
    for i, j in user_item_train.items():
        user_id = np.int32(user['index'][i]['user_id'])
        sku_id = np.int32(product['index'][j]['sku_id'])

        table.append([
            np.int32(user_item_label[i][j]),
            np.int32(user_id),
            np.int32(sku_id),

            np.int32(user_item_action_[1][i][j]),
            np.int32(user_item_action_[2][i][j]),
            np.int32(user_item_action_[3][i][j]),
            np.int32(user_item_action_[4][i][j]),
            np.int32(user_item_action_[5][i][j]),
            np.int32(user_item_action_[6][i][j]),

            np.int32(user['index'][i]['sex']),
            np.int32(user['index'][i]['age']),
            np.int32(user['index'][i]['user_lv_cd']),
            np.int32(user['index'][i]['user_reg_tm']),

            np.int32(product['index'][j]['a1']),
            np.int32(product['index'][j]['a2']),
            np.int32(product['index'][j]['a3']),
            np.int32(product['index'][j]['cate']),
            np.int32(product['index'][j]['brand']),

            # np.int(user_a_[1][i][j]),
            # np.int(user_a_[2][i][j]),
            # np.int(user_a_[3][i][j]),

            ])

    df = pd.DataFrame(table, columns=columns)

    feats = [pd.get_dummies(df[col], prefix=col) for col in ['user_sex', 'user_age', 'user_lv_cd', 'user_reg_tm', 'sku_a1', 'sku_a2', 'sku_a3']]
    df = pd.concat([df[['label', 'user_id', 'sku_id', 'act_1', 'act_2', 'act_3', 'act_4', 'act_5', 'act_6']], feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6]], axis=1)
    # df = pd.concat([df[['label', 'user_id', 'sku_id', 'act_1', 'act_2', 'act_3', 'act_4', 'act_5', 'act_6']], feats[0], feats[1], feats[2], feats[3], feats[4], feats[5], feats[6], feats[7], feats[8], feats[9]], axis=1)
    # feats = [pd.get_dummies(df[col], prefix=col) for col in ['user_sex', 'user_age', 'user_lv_cd', 'user_reg_tm', 'sku_a1', 'sku_a2', 'sku_a3', 'user_a1', 'user_a2', 'user_a3']]

    df.to_csv(train_path % (d1, d2, d3, d4), index=False)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage:')
        print('\tipython3 gen_feat.py 20160313')
        print('\tipython3 gen_feat.py 20160318')
    else:
        # d1 ~ d2 训练数据 d3 ~ d4标签
        # d1 = '20160313'
        # d1 = '20160318'
        d1 = sys.argv[1]
        d2 = ndays_after(28, d1)
        d3 = ndays_after(1, d2)
        d4 = ndays_after(4, d3)

        make_train_data(d1, d2, d3, d4)
