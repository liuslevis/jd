from gen_feat import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb

train_path = 'data/input/train_%s_%s_%s_%s.csv'
model_path = 'data/output/bst.model'
submission_path = 'data/output/submission.csv'

FALSE_TRUE_SAMPLE_RATE = 32 / 1

threshold = 0.5
missing_value = -999.0
labels = ['0', '1']

IGNORE_FEATS = [
    'brand',
]
SEL_FEATS = [
    # 'act2',
    # 'user_act1_inteval0',
    # 'user_brand36_inteval0',
    # 'act_5',
    # 'user_cat8',
    # 'sku_a3',
    # 'user_brand36_inteval1',
    # 'user_act0_brand36_inteval5',
    # 'act_1',
    # 'act3',
    # 'user_age_3',
    # 'user_act5_brand55_inteval5',
    # 'user_brand48_inteval0',
    # 'user_reg_tm',
    # 'user_act5_brand59_inteval5',
    # 'user_cat8_inteval5',
    # 'act_6',
    # 'user_prop3_act1_inteval0',
    # 'user_act0_inteval1',
    ]

def select_feats(df):
    ignores = []
    for ignore in IGNORE_FEATS:
        for col in df.columns:
            if ignore in col:
                ignores.append(col)
    # return list(set(ret))
    return list(set(df.columns) - set(['label'] + ignores))

def strip_id(df):
    sel_cols = list(set(df.columns) - set(['user_id', 'sku_id']))
    return df[sel_cols]

def read_input_data(d1):
    d2 = ndays_after(train_days, d1)
    d3 = ndays_after(1, d2)
    d4 = ndays_after(label_days, d3)
    return pd.read_csv(train_path % (d1, d2, d3, d4))

def read_train_combi(d1_li):
    print('\ncombi:', ' '.join(d1_li))
    combi = pd.concat([read_input_data(d1) for d1 in d1_li])
    combi_true = combi[combi['label']==1]
    combi_false = combi[combi['label']==0]
    false_num = int(len(combi_true) * FALSE_TRUE_SAMPLE_RATE)
    combi = pd.concat([combi_true, combi_false[:false_num]])
    return combi

# F11,F12,score
def report(X, y, y_pred, print_score=False):
    y = y.values

    true_users = set()
    pred_users = set()
    all_users  = set()
    true_records = [] # [(user_id,item_id),...]
    pred_records = []
    all_records  = []
    for i,row in X[['user_id','sku_id']].iterrows():
        user_id, item_id = row['user_id'], row['sku_id']
        all_users.add(user_id)
        all_records.append((user_id, item_id))
        if y[i] == 1:
            true_records.append((user_id, item_id))
            true_users.add(user_id)
        if y_pred[i]==1:
            pred_records.append((user_id, item_id))
            pred_users.add(user_id)

    if len(pred_users) == 0:
        print('no buy prediction!')
        return

    user_cm = np.zeros((2,2))
    for user in all_users:
        i = int(user in true_users)
        j = int(user in pred_users)
        user_cm[i][j] += 1
    # print_cm(user_cm, labels)
    user_recall = user_cm[1][1] / (user_cm[1][1] + user_cm[0][1]) # TP / (TP+FN)
    user_acc    = user_cm[1][1] / (user_cm[1][1] + user_cm[0][0]) # TP / (TP+FP)

    record_cm = np.zeros((2,2))
    for record in all_records:
        i = int(record in true_records)
        j = int(record in pred_records)
        record_cm[i][j] += 1
    # print_cm(record_cm, labels)
    record_recall = record_cm[1][1] / (record_cm[1][1] + record_cm[0][1]) # TP / (TP+FN)
    record_acc    = record_cm[1][1] / (record_cm[1][1] + record_cm[0][0]) # TP / (TP+FP)

    F11 = 6.0 * user_recall * user_acc / (5.0 * user_recall + user_acc)
    F12 = 5.0 * record_acc * record_recall / (2.0 * record_recall + 3 * record_acc)
    score = 0.4 * F11 + 0.6 * F12
    if print_score:
        print('F11 = %.4f' % F11)
        print('F12 = %.4f' % F12)
        print('score =  %.4f' % score)
    return score, F11, F12

def make_submission(d1, submission_path):
    combi = read_input_data(d1)
    features = select_feats(combi)
    X = combi[features]
    y = combi['label']
    data = xgb.DMatrix(strip_id(X), label=y, missing = missing_value)
    
    bst = xgb.Booster({'nthread':4})
    bst.load_model(model_path)    
    y_score = bst.predict(data)
    y_pred = np.int32(y_score > threshold)

    pred_records = {} #{user_id:sku_id}
    for i,row in X[['user_id','sku_id']].iterrows():
        user_id, item_id = row['user_id'], row['sku_id']
        score = y_score[i]
        if y_pred[i]==1:
            if user_id not in pred_records or score > pred_records[user_id]:
                pred_records.update({user_id:item_id})

    df = pd.DataFrame(list(pred_records.items()), columns=['user_id','sku_id'])
    df.to_csv(submission_path, index=False)
    print('\nsubmission\tpath')
    print('%s\t%s' %(d1, submission_path))
    return df

def validate(d1_li, bst=None, print_cm=False):
    print('validate\tscore\tF11\tF12')
    for d1 in d1_li:
        combi = read_input_data(d1)
        features = select_feats(combi)
        X_valid = combi[features]
        y_valid = combi['label']
        d_valid = xgb.DMatrix(strip_id(X_valid), label=y_valid, missing = missing_value)

        if bst is None:
            bst = xgb.Booster({'nthread':4})
            bst.load_model(model_path)

        y_valid_score = bst.predict(d_valid)
        y_valid_pred = np.int32(y_valid_score > threshold)

        if print_cm:
            cm = confusion_matrix(y_valid, y_valid_pred)
            print_cm(cm, labels)

        score, F11, F12 = report(X_valid, y_valid, y_valid_pred, print_score=False)
        print('%s\t%.4f\t%.4f\t%.4f' % (d1, score, F11, F12))

def train(combi, print_cm=False):
    print('\ntrain:')
    features = select_feats(combi)
    X_combi = combi[features]
    y_combi = combi['label']
    X_train, X_test, y_train, y_test = train_test_split(X_combi, y_combi, test_size=0.5, random_state=0)

    params = {
        'n_estimators':500,
        'max_depth':8,
        'eta':0.1, 
        'silent':1, 
        'objective':'binary:logistic', 
        'nthread':4, 
        'eval_metric':['auc', 'logloss', 'error']
        }

    print('samples: %d/%d' % (len(y_train), len(y_test)))
    print('sel feats:', ' '.join(features))
    # print('using  feats:', ' '.join(features))
    print('param:', params)

    d_train = xgb.DMatrix(strip_id(X_train), label=y_train, missing = missing_value)
    d_test = xgb.DMatrix(strip_id(X_test), label=y_test, missing = missing_value)
    evallist = [(d_test, 'eval'), (d_train, 'train')]
    num_round = 5
    bst = xgb.train(params, d_train, num_round, evallist)
    bst.save_model(model_path)
    xgb.plot_importance(bst)
    y_test_pred = np.int32(bst.predict(d_test) > threshold)
    if print_cm:
        cm = confusion_matrix(y_test, y_test_pred)
        print_cm(cm, labels)
    return bst

combi = read_train_combi(['2016%04d' % i for i in range(201,212)])
bst = train(combi)
validate(['2016%04d' % i for i in [221,226]], bst)

# bst = train(read_train_combi(['2016%04d' % i for i in [201]]))
# make_submission('20160318', submission_path)

import matplotlib.pyplot as plt 
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 48, 16
plt.style.use('ggplot') 
xgb.plot_importance(bst) 
xgb.plot_tree(bst, num_trees=1) 
xgb.to_graphviz(bst, num_trees=1)
# plt.savefig('data/output/xgb.png')  
plt.show()