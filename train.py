from gen_feat import *
from sklearn.model_selection import train_test_split
import xgboost as xgb

d1 = '20160201'
d2 = '20160214'
d3 = '20160215'
d4 = '20160219'

input_path = 'data/input/train_%s_%s_%s_%s.csv' % (d1, d2, d3, d4)

combi = pd.read_csv(input_path)
features = list(set(combi.columns) - set('label'))
X_combi = combi[features].values
y_combi = combi['label'].values
X_train, X_test, y_train, y_test = train_test_split(X_combi, y_combi, test_size=0.2, random_state=0)


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

y_test_pred = bst.predict(dtest)
y_test

report(y_test_pred, y_test)


# def report(pred, label):

#     actions = label
#     result = pred

#     # 所有用户商品对
#     all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
#     all_user_item_pair = np.array(all_user_item_pair)
#     # 所有购买用户
#     all_user_set = actions['user_id'].unique()

#     # 所有品类中预测购买的用户
#     all_user_test_set = result['user_id'].unique()
#     all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
#     all_user_test_item_pair = np.array(all_user_test_item_pair)

#     # 计算所有用户购买评价指标
#     pos, neg = 0,0
#     for user_id in all_user_test_set:
#         if user_id in all_user_set:
#             pos += 1
#         else:
#             neg += 1
#     all_user_acc = 1.0 * pos / ( pos + neg)
#     all_user_recall = 1.0 * pos / len(all_user_set)
#     print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
#     print('所有用户中预测购买用户的召回率' + str(all_user_recall))

#     pos, neg = 0, 0
#     for user_item_pair in all_user_test_item_pair:
#         if user_item_pair in all_user_item_pair:
#             pos += 1
#         else:
#             neg += 1
#     all_item_acc = 1.0 * pos / ( pos + neg)
#     all_item_recall = 1.0 * pos / len(all_user_item_pair)
#     print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
#     print('所有用户中预测购买商品的召回率' + str(all_item_recall))
#     F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
#     F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
#     score = 0.4 * F11 + 0.6 * F12
#     print('F11=' + str(F11))
#     print('F12=' + str(F12))
#     print('score=' + str(score))

if __name__ == '__main__':
    xgboost_cv()
    # xgboost_make_submission()