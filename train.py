from gen_feat import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt 

train_path = 'data/input/train_%s_%s_%s_%s.csv'
model_path = 'data/output/bst.model'
submission_path = 'data/output/submission.csv'

threshold = 0.5
missing_value = -999.0
labels = ['0', '1']
ignore_feats = [
    # 'user_a1', 
    # 'user_a2', 
    # 'user_a3',
    ]

def get_feat(df, ignore_feats):
    ignores = []
    for prefix in ignore_feats:
        for col in df.columns:
            if col.startswith(prefix):
                ignores.append(col)
    return list(set(df.columns) - set(['label'] + ignores))

def strip_id(df):
    sel_cols = list(set(df.columns) - set(['user_id', 'sku_id']))
    return df[sel_cols]

def read_input_data(d1):
    d2 = ndays_after(train_days, d1)
    d3 = ndays_after(1, d2)
    d4 = ndays_after(label_days, d3)
    return pd.read_csv(train_path % (d1, d2, d3, d4))

def train(d1, print_cm=False):
    print('\ntrain')
    print(d1)
    combi = read_input_data(d1)
    features = get_feat(combi, ignore_feats)
    combi_true = combi[combi['label']==1]
    combi_false = combi[combi['label']==0]
    combi = pd.concat([combi_true, combi_false[:len(combi_true)]])
    X_combi = combi[features]
    y_combi = combi['label']
    X_train, X_test, y_train, y_test = train_test_split(X_combi, y_combi, test_size=0.5, random_state=0)

    d_train = xgb.DMatrix(strip_id(X_train), label=y_train, missing = missing_value)
    d_test = xgb.DMatrix(strip_id(X_test), label=y_test, missing = missing_value)
    params = {
        'max_depth':2, 
        'eta':0.05, 
        'silent':1, 
        'objective':'binary:logistic', 
        'nthread':4, 
        'eval_metric':['auc', 'logloss']
        }
    evallist = [(d_test, 'eval'), (d_train, 'train')]
    num_round = 2
    bst = xgb.train(params, d_train, num_round, evallist)
    bst.save_model(model_path)
    xgb.plot_importance(bst)
    y_test_pred = np.int32(bst.predict(d_test) > threshold)
    if print_cm:
        cm = confusion_matrix(y_test, y_test_pred)
        print_cm(cm, labels)
    return bst

# F11,F12,score
def report(X, y, y_pred, print_score=False):
    y = y.values

    true_records = [] # [(user_id,item_id),...]
    pred_records = []

    true_users = set()
    pred_users = set()


    for i,row in X[['user_id','sku_id']].iterrows():
        user_id, item_id = row['user_id'], row['sku_id']
        if y[i] == 1:
            true_records.append((user_id, item_id))
            true_users.add(user_id)
        if y_pred[i]==1:
            pred_records.append((user_id, item_id))
            pred_users.add(user_id)

    if len(pred_users) == 0:
        print('no buy prediction!')
        return

    all_users = pred_users.union(true_users)
    all_records = pred_records + true_records

    pos = 0
    neg = 0
    for pred_user in pred_users:
        if pred_user in true_users:
            pos += 1
        else:
            neg += 1
    user_recall = 1.0 * pos / len(all_users)
    user_acc = 1.0 * pos / ( pos + neg)

    pos = 0
    neg = 0
    for pred_record in pred_records:
        if pred_record in true_records:
            pos += 1
        else:
            neg += 1
    record_recall = 1.0 * pos / len(pred_records)
    record_acc = 1.0 * pos / ( pos + neg)

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
    features = get_feat(combi, ignore_feats)
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

def validate(d1_li, print_cm=False):
    print('validate\tscore\tF11\tF12')
    for d1 in d1_li:
        combi = read_input_data(d1)
        features = get_feat(combi, ignore_feats)
        X_valid = combi[features]
        y_valid = combi['label']
        d_valid = xgb.DMatrix(strip_id(X_valid), label=y_valid, missing = missing_value)

        bst = xgb.Booster({'nthread':4})
        bst.load_model(model_path)
        y_valid_score = bst.predict(d_valid)
        y_valid_pred = np.int32(y_valid_score > threshold)

        if print_cm:
            cm = confusion_matrix(y_valid, y_valid_pred)
            print_cm(cm, labels)

        score, F11, F12 = report(X_valid, y_valid, y_valid_pred, print_score=False)
        print('%s\t%.4f\t%.4f\t%.4f' % (d1, score, F11, F12))


# for d1 in range(201, 205):
#     bst = train('2016%04d' % d1)
# validate(['2016%04d' % i for i in range(d1+5, d1+8)])

bst = train('20160201')
validate(['20160206'])

# bst = train('20160327')
# validate(['20160401'])

# make_submission('20160318', submission_path)


# plt.style.use('ggplot') 
# xgb.plot_importance(bst) 
# xgb.plot_tree(bst, num_trees=1) 
# xgb.to_graphviz(bst, num_trees=1)
# plt.show()

if __name__ == '__main__':
    pass
