import time
import pickle
from datetime import datetime, timedelta
from get_feat import make_set, report, date_change
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import pandas as pd


LENGTH = 31
NUM_ROUND = 100
LABEL_BOUND = 0.06


def set_split(end_date, length=LENGTH, is_train=True, is_cate8=False):
    start_date = date_change(end_date, -length)
    print(start_date, end_date)
    return make_set(start_date, end_date, is_train, is_cate8)


def xgboost_model(end_date, num_round=NUM_ROUND, num=1):
    # date = end_date
    # train_data = None
    # label = None
    # for _ in range(num):
    #     _, temp_train_data, temp_label = set_split(date)
    #     if train_data is None and label is None:
    #         train_data = temp_train_data
    #         label = temp_label
    #     else:
    #         train_data = pd.concat([train_data, temp_train_data], axis=0)
    #         label = pd.concat([label, temp_label], axis=0)
    #     date = date_change(date, -5)
    _, train_data, label = set_split(end_date)
    x_train, x_test, y_train, y_test = train_test_split(
        train_data.values, label.values, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    # dtrain.feature_names = list(train_data.columns)
    # dtest.feature_names = list(train_data.columns)
    param = {
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'max_depth': 3,
        # 'max_delta_step': 0,
        'min_child_weight': 3,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,
        'eta': 0.05,
        'silent': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'nthread': 16
    }
    # param['eval_metric'] = "auc"
    plst = list(param.items())
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(plst, dtrain, num_round, evallist)
    print('训练完成')
    return bst


def xgboost_result(train_end_date, pred_end_date, bst=None):
    if bst is None:
        bst = xgboost_model(train_end_date)

    pred_user_index, pred_train_data = set_split(
        pred_end_date, is_train=False, is_cate8=True)
    pred_result = bst.predict(xgb.DMatrix(pred_train_data.values))
    pred = pred_user_index.copy()
    pred['label'] = pred_result
    # pred['label'] = pred['label'].map(lambda x: 1 if x >= LABEL_BOUND else 0)
    pred = pred[pred['label'] >= LABEL_BOUND]
    pred = pred.sort_values(by='label', ascending=False).groupby(
        'user_id').first().reset_index()
    pred = pred[['user_id', 'sku_id']]
    pred['user_id'] = pred['user_id'].astype(int)
    result_name = 'result/submission_%s.csv' % time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime())
    pred.to_csv(result_name, index=False, index_label=False)
    print('成功生成预测结果')


def xgboost_test(train_end_date, test_pred_end_date, bst=None):
    if bst is None:
        bst = xgboost_model(train_end_date)

    test_pred_user_index, test_pred_train_data, test_pred_label = set_split(
        test_pred_end_date, is_cate8=True)
    fact = test_pred_user_index.copy()
    fact['label'] = test_pred_label
    test_pred_result = bst.predict(xgb.DMatrix(test_pred_train_data.values))
    pred = test_pred_user_index.copy()
    pred['label'] = test_pred_result
    pred = pred.sort_values(by='label', ascending=False).groupby(
        'user_id').first().reset_index()
    pred['label'] = pred['label'].map(lambda x: 1 if x >= LABEL_BOUND else 0)
    pickle.dump(pred, open('cache/pred_%s.pkl' % (test_pred_end_date), 'wb'))

    # report(pred, fact)
    report(pred, pred_end_date=test_pred_end_date)


if __name__ == '__main__':
    train_end_date = '2016-04-05'
    test_pred_end_date = '2016-04-10'
    pred_end_date = '2016-04-16'
    # for k in reversed(range(7)):
    #     try:
    #         print(k)
    #         bst = xgboost_model(train_end_date, num=k)
    #         xgboost_test(train_end_date, test_pred_end_date, bst=bst)
    #         bst = xgboost_model(test_pred_end_date, num=k)
    #         xgboost_result(test_pred_end_date, pred_end_date, bst=bst)
    #         break
    #     except:
    #         continue
    xgboost_test(train_end_date, test_pred_end_date)
    # xgboost_result(test_pred_end_date, pred_end_date)
