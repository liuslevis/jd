#TODO 测试集要用训练集的 User.all * Product.all
from gen_feat import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import xgboost as xgb

# d1 = '20160201'
# d2 = '20160229'
# d3 = '20160301'
# d4 = '20160305'

d1 = '20160315'
d2 = '20160415'
d3 = '20160416'
d4 = '20160420'

input_path = 'data/input/train_%s_%s_%s_%s.csv' % (d1, d2, d3, d4)

def strip_ids(df):
    return df[list(set(df.columns) - set(['user_id', 'sku_id']))]

def report(X, features, y_true, y_pred, threshold=0.5):
    assert X.shape[0]==y.shape[0]==y_pred.shape[0], 'rows of X, y, y_pred not the same'
    y_pred = np.int32(y_pred > threshold)
    y_true = y_true.values
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(X.shape[0]):
        row = X.iloc[[i]]
        user_id = row['user_id'].values[0]
        sku_id = row['sku_id'].values[0]
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 0
    precise = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)

    # f11 = 6 * recall * precise / (5 * recall + precise)
    # f12 = 5 * recall * precise / (2 * recall + 3 * precise)
    # score = f11 * 0.4 + f12 * 0.6
    print('\tTP\tTN\n\t%d\t%d\nFP\t%d\t%d\nFN' % (tp, tn, fp, fn))
    print('precise:%.4f\trecall:%.4f' % (precise, recall))
    # print('f1:%.4f\tf2:%.4f' % (f11, f12))
    # print('score:%.4f' % (score))

# Train
combi = pd.read_csv(input_path)
features = list(set(combi.columns) - set('label'))
X_combi = combi[features]
y_combi = combi['label']
X_train, X_test, y_train, y_test = train_test_split(X_combi, y_combi, test_size=0.2, random_state=0)


dtrain=xgb.DMatrix(strip_ids(X_train), label=y_train)
dtest=xgb.DMatrix(strip_ids(X_test), label=y_test)
param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = "auc"
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 10
bst=xgb.train(plst, dtrain, num_round, evallist)

y_test_pred = bst.predict(dtest)
y_test

report(X_test, features, y_test, y_test_pred, threshold=0.5)

# Submission: TODO all same value
X_submit = pd.read_csv(input_path)[features]
dsubmit = xgb.DMatrix(strip_ids(X_submit))
y_sub = bst.predict(dsubmit)


if __name__ == '__main__':
    xgboost_cv()
    # xgboost_make_submission()