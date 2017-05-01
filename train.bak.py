__author__ = 'foursking'
from gen_feat import make_train_set
from gen_feat import make_submission_set
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_feat import report

def xgboost_make_submission():
    X_train_start_date = '2016-03-10'
    X_train_end_date = '2016-04-11'
    y_train_start_date = '2016-04-11'
    y_train_end_date = '2016-04-16'

    X_test_start_date = '2016-03-15'
    y_test = '2016-04-16'

    user_index, training_data, label = make_train_set(X_train_start_date, X_train_end_date, y_train_start_date, y_train_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    #param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_data = make_submission_set(X_test_start_date, y_test,)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('./sub/submission.csv', index=False, index_label=False)



def xgboost_cv():
X_train_start_date = '2016-03-05'
X_train_end_date = '2016-04-11'
y_train_start_date = '2016-04-11'
y_train_end_date = '2016-04-16'

# submission
X_test_start_date = '2016-02-05'
X_test_end_date = '2016-03-05'
y_test_start_date = '2016-03-05'
y_test_end_date = '2016-03-10'

train_user_index, train_data, train_label = make_train_set(X_train_start_date, X_train_end_date, y_train_start_date, y_train_end_date)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=0)
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)
param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = "auc"
plst = list(param.items())
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 10
bst=xgb.train(plst, dtrain, num_round, evallist)

test_user_index, test_data, test_label = make_train_set(X_test_start_date, X_test_end_date, y_test_start_date, y_test_end_date)
test = xgb.DMatrix(test_data)
y = bst.predict(test)

pred = test_user_index.copy()
y_true = test_user_index.copy()
pred['label'] = y
y_true['label'] = label
report(pred, y_true)


if __name__ == '__main__':
    xgboost_cv()
    # xgboost_make_submission()