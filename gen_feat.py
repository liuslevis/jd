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
from itertools import combinations, permutations

train_days = 28 # 14
label_days = 4

action_paths = './data/raw/JData_Action_%s.csv' 
comment_path = './data/raw/JData_Comment.csv'
product_path = './data/raw/JData_Product.csv'
user_path    = './data/raw/JData_User.csv'
train_path   = 'data/input/train_%s_%s_%s_%s.csv'

USE_CACHE = True
ACTION_TYPES = 6

comment_date = ['20160201', '20160208', '20160215', '20160222', '20160229', '20160307', '20160314', '20160321', '20160328', '20160404', '20160411', '20160415']

brands_li = [3,13,14,24,25,30,48,49,51,70,76,83,88,90,91,101,116,124,127,159,174,180,197,200,209,211,214,225,227,244,249,263,283,285,291,299,306,318,321,324,328,331,336,354,355,370,375,383,403,404,427,438,453,479,484,489,499,515,541,545,554,556,561,562,571,574,594,596,599,605,622,623,635,655,658,665,673,674,677,693,717,739,752,759,766,772,790,800,801,804,812,837,855,857,871,875,885,900,905,907,916,922,]
brands = {brand : brands_li.index(brand) for brand in brands_li}
user_brand_cols = ['user_brand_%d' % i for i in range(len(brands))]

def recent_comment_date(date, comment_date=comment_date):
    for each in comment_date:
        if each < date:
            continue
        else:
            return each
    return comment_date[0]


def ndays_after(ndays, date_str):
    return datetime.strftime(datetime.strptime(date_str, '%Y%m%d') + timedelta(days=ndays), '%Y%m%d')

def strptime(dt_str):
    return datetime.strptime(dt_str.replace('-', ''), '%Y%m%d')

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

def convert_month_stage(dt):
    day = int(dt) % 100
    if day < 10:
        return 0
    elif day < 20:
        return 1
    else:
        return 2

def month_stage_between(d1, d2):
    dt1 = datetime.strptime(d1, '%Y%m%d')
    dt2 = datetime.strptime(d2, '%Y%m%d')
    dt_mid = datetime.strftime(dt1 + (dt2 - dt2) / 2, '%Y%m%d')
    return convert_month_stage(dt_mid)

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
        return -2

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
    return pd.read_csv(user_path)

def get_product_df():
    return pd.read_csv(product_path)

def get_comment_df():
    return pd.read_csv(comment_path)

def get_user(d1, d2, d3, d4):    
    cache_path = './cache/user_%s.pkl' % (d4)
    if os.path.exists(cache_path) and USE_CACHE:
        return pickle.load(open(cache_path, 'rb'))
    else:
        df = get_user_df(d1, d2, d3, d4)        
        df['age'] = df['age'].map(convert_age)
        df['sex'] = df['sex'].fillna(-1)
        df['user_reg_tm'] = df['user_reg_tm']\
            .map(lambda reg_tm : (strptime(d4) - strptime(reg_tm)).days if type(reg_tm) is str else -1)\
            .map(convert_reg_tm)
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
        pickle.dump(ret, open(cache_path, 'wb'))
        return ret

#{'sku_id':{sku_id:info}, 'index':{index:info}}
def get_product():
    cache_path = './cache/product.pkl'
    if os.path.exists(cache_path) and USE_CACHE:
        return pickle.load(open(cache_path, 'rb'))
    else:
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
        pickle.dump(ret, open(cache_path, 'wb'))
        return ret

# {sku_id:{'20160201':{comment_num:x,has_bad_comment:x,bad_comment_rate:x}}}
def get_comment():
    cache_path = './cache/comment.pkl'
    if os.path.exists(cache_path) and USE_CACHE:
        return pickle.load(open(cache_path, 'rb'))
    else:
        df = get_comment_df()
        cols = ['comment_num', 'has_bad_comment', 'bad_comment_rate']
        ret = {}
        for index, row in df.iterrows():
            sku_id = df.ix[index, 'sku_id']
            date = df.ix[index, 'dt'].replace('-', '')
            info = {col:df.ix[index, col] for col in cols}
            ret.update({sku_id:{date:info}})
        pickle.dump(ret, open(cache_path, 'wb'))
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

# d1 = '20160206'
# d2 = ndays_after(train_days, d1)
# d3 = ndays_after(1, d2)
# d4 = ndays_after(4, d3)
def make_train_data(d1, d2, d3, d4):
    user = get_user(d1, d2, d3, d4)
    product = get_product()
    comment = get_comment()

    user_set = set(user['user_id'].keys())
    sku_set = set(product['sku_id'].keys())

    user_len = len(user_set)
    product_len = len(sku_set)
    brand_len = len(brands)

    user_item_train = {} # {i:j}
    user_item_label = np.zeros((user_len, product_len)) # M[user_index][item_index] = label
    user_item_action_ = np.zeros((ACTION_TYPES+1, user_len, product_len)) # M[type][user_index][item_index] = sum

    user_ai_ = None #M[a_i][user_index]
    user_ai_pos = np.ones((3+1, user_len), dtype=np.int32)
    user_ai_neg = np.ones((3+1, user_len), dtype=np.int32)

    user_cat8     = None
    user_cat8_pos = np.ones((user_len), dtype=np.int32)
    user_cat8_neg = np.ones((user_len), dtype=np.int32)

    user_brand_ = np.ones((brand_len, user_len), dtype=np.int32)

    user_buy_month_stage_  = np.zeros((3, user_len), dtype=np.int32)

    user_pred_month_stage_ = np.zeros(3, dtype=np.int32)
    user_pred_month_stage_[month_stage_between(d3, d4)] = 1

    dates = list(set(map(lambda d:d[:-2], [d1, d2, d3, d4])))
    for date in dates:
        action_path = action_paths % date
        if not os.path.exists(action_path):
            continue
        with open(action_path) as f:
            for line in f.readlines():
                if line.startswith('user_id,sku_id,time,model_id,type,cate,brand'):
                    continue
                user_id, sku_id, time, model_id, type_, cate, brand = parse_action_line(line)
                date = time.split(' ')[0].replace('-', '')
                
                if date > d4:
                    break

                # record in d1~d4
                if d1 <= date <= d4 and sku_id in sku_set:
                    i = user['user_id'][user_id]['index']
                    j = product['sku_id'][sku_id]['index']
                    
                    month_stage = convert_month_stage(date)

                    # train d1~d2
                    if d1 <= date <= d2 and 1 <= type_ <= 6:
                        if type_ >= user_item_action_.shape[0] or i >= user_item_action_.shape[1] or j >= user_item_action_.shape[2]:
                            print('debug', user_item_action_.shape, type_, i, j)

                        user_item_action_[type_][i][j] += 1
                        user_item_train.update({i:j})

                        # user_a1 2 3
                        for k in range(1,4):
                            ai = 'a%d' % k
                            if product['index'][j][ai] > -1:
                                user_ai_pos[k][i] += 1 
                            else:
                                user_ai_neg[k][i] += 1

                        # user_cat8
                        if cate == 8:
                            user_cat8_pos[i] += 1
                        else:
                            user_cat8_neg[i] += 1

                        # user_brand
                        k = brands[brand]
                        user_brand_[k][i] += 1

                        # if type_==4: TODO
                        # user_buy_month_stage_0~3
                        user_buy_month_stage_[month_stage][i] += 1

                    # label d3~d4. !!DONT GET FEAT FROM HERE!!
                    if d3 <= date <= d4 and type_ == 4:
                        user_item_label[i][j] = 1
                        user_item_train.update({i:j})
                        

                    
    user_cat8 = np.float64(user_cat8_pos / (user_cat8_pos + user_cat8_neg))
    user_ai_  = np.float64(user_ai_pos   / (user_ai_pos + user_ai_neg))
    user_brand_ = user_brand_ / user_brand_.sum(axis=0) # normalize

    comment_date = '20160201'

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

            'sku_comment_num',
            'sku_has_bad_comment',
            'sku_bad_comment_rate',

            'user_a1',
            'user_a2',
            'user_a3',
            'user_cat8', #0~1

            'user_buy_month_stage_0',
            'user_buy_month_stage_1',
            'user_buy_month_stage_2',
            'user_pred_month_stage_0',
            'user_pred_month_stage_1',
            'user_pred_month_stage_2',

        ] + user_brand_cols # user_brand_[1] = 0.x

    table = []
    for i, j in user_item_train.items():
        user_id = np.int32(user['index'][i]['user_id'])
        sku_id = np.int32(product['index'][j]['sku_id'])
        comment_info = comment[sku_id][comment_date] if sku_id in comment and comment_date in comment[sku_id] else {'comment_num':0, 'has_bad_comment':-1, 'bad_comment_rate':0}
        
        table.append([
            np.int32(user_item_label[i][j]),
            np.int32(user_id),
            np.int32(sku_id),]
            +
            [np.int32(user_item_action_[k][i][j]) for k in [1,2,3,4,5,6]]
            +
            [np.int32(user['index'][i]['sex']),
            np.int32(user['index'][i]['age']),
            np.int32(user['index'][i]['user_lv_cd']),
            np.int32(user['index'][i]['user_reg_tm']),]
            +
            [np.int32(product['index'][j]['a1']),
            np.int32(product['index'][j]['a2']),
            np.int32(product['index'][j]['a3']),
            np.int32(product['index'][j]['cate']),
            np.int32(product['index'][j]['brand']),]
            +
            [np.int32( comment_info['comment_num']),
            np.int32(  comment_info['has_bad_comment']),
            np.float64(comment_info['bad_comment_rate'])]
            +
            [np.int32(user_ai_[k][i]) for k in [1,2,3]]
            +
            [np.float64(user_cat8[i]),] 
            +
            [user_buy_month_stage_[k][i] for k in range(3)]
            +
            [user_pred_month_stage_[k] for k in range(3)] 
            +
            [user_brand_[k][i] for k in range(len(brands))]) 

    df = pd.DataFrame(table, columns=columns)
    dummy_feats = [pd.get_dummies(df[col], prefix=col) for col in ['user_sex', 'user_age']]
    df = pd.concat([df[['label', 'user_id', 'sku_id', 'act_1', 'act_2', 'act_3', 'act_4', 'act_5', 'act_6', 'user_lv_cd', 'user_reg_tm', 'user_a1', 'user_a2', 'user_a3', 'user_cat8', 'sku_a1', 'sku_a2', 'sku_a3','sku_comment_num', 'sku_has_bad_comment', 'sku_bad_comment_rate', 'user_buy_month_stage_0', 'user_buy_month_stage_1', 'user_buy_month_stage_2', 'user_pred_month_stage_0', 'user_pred_month_stage_1', 'user_pred_month_stage_2'] + user_brand_cols]] + dummy_feats, axis=1)

    path = train_path % (d1, d2, d3, d4)
    df.to_csv(path, index=False, float_format='%.6f')
    print(path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage:')
        print('\t28')
        print('\tipython3 gen_feat.py 20160313 20160318')
        print('\t14')
        print('\tipython3 gen_feat.py 20160327 20160401')
        
    else:
        # d1 ~ d2 训练数据 d3 ~ d4标签
        for d1 in sys.argv[1:]:
            d2 = ndays_after(train_days, d1)
            d3 = ndays_after(1, d2)
            d4 = ndays_after(label_days, d3)
            make_train_data(d1, d2, d3, d4)
