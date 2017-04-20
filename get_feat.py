#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pickle
import os
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

action_base_path = "data/JData_Action_20160"
comment_path = "data/JData_Comment.csv"
product_path = "data/JData_Product.csv"
user_path = "data/JData_User.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22",
                "2016-02-29", "2016-03-07", "2016-03-14", "2016-03-21",
                "2016-03-28", "2016-04-04", "2016-04-11", "2016-04-15"]


def part_read_csv(filename, chunksize=1000000, func=None, func_para=None, **argv):
    chunks = []
    reader = pd.read_csv(filename, iterator=True, **argv)
    while True:
        try:
            chunk = reader.get_chunk(chunksize)
            if func:
                chunk = func(chunk, **func_para)
            if len(chunk):
                chunks.append(chunk)
        except StopIteration:
            break
    temp_df = pd.concat(chunks, ignore_index=True) if chunks else chunk
    print('读取完成')
    return temp_df


def date_change(before_date, change):
    after_date = (datetime.strptime(
        before_date, '%Y-%m-%d') + timedelta(days=change))
    return after_date.strftime('%Y-%m-%d')


def get_basic_user_feat():
    dump_path = 'cache/basic_user.pkl'
    if os.path.exists(dump_path):
        user = pickle.load(open(dump_path, 'rb'))
    else:
        user = pd.read_csv(user_path, encoding='gbk')
        age_list_before = ['-1', '15岁以下', '16-25岁',
                           '26-35岁', '36-45岁', '46-55岁', '56岁以上']
        age_list_after = [-1, 0, 1, 2, 3, 4, 5]
        user['age'].replace(age_list_before, age_list_after, inplace=True)
        age_df = pd.get_dummies(user["age"], prefix="age")
        sex_df = pd.get_dummies(user["sex"], prefix="sex")
        user_lv_df = pd.get_dummies(user["user_lv_cd"], prefix="user_lv_cd")
        user = pd.concat([user['user_id'], age_df, sex_df, user_lv_df], axis=1)
        pickle.dump(user, open(dump_path, 'wb'))
    return user


def get_basic_product_feat():
    dump_path = 'cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path, 'rb'))
    else:
        product = part_read_csv(product_path)
        attr1_df = pd.get_dummies(product["a1"], prefix="a1")
        attr2_df = pd.get_dummies(product["a2"], prefix="a2")
        attr3_df = pd.get_dummies(product["a3"], prefix="a3")
        product = pd.concat(
            [product[['sku_id', 'cate', 'brand']], attr1_df, attr2_df, attr3_df], axis=1)
        pickle.dump(product, open(dump_path, 'wb'))
    return product


def get_comments_product_feat(end_date):
    dump_path = 'cache/comments_accumulate_%s.pkl' % (end_date)
    if os.path.exists(dump_path):
        comments = pickle.load(open(dump_path, 'rb'))
    else:
        comments = part_read_csv(comment_path)
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin)
                            & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1)  # type: pd.DataFrame
        #del comments['dt']
        #del comments['comment_num']
        comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate',
                             'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
        pickle.dump(comments, open(dump_path, 'wb'))
    return comments


def filter_date(action, start_date, end_date):
    return action[(action.time >= start_date) & (action.time < end_date)]


def get_action(i, start_date, end_date):
    action_path = action_base_path + "%d.csv" % (i + 1)
    dump_path = 'cache/action_%d_%s_%s.pkl' % (i, start_date, end_date)
    if os.path.exists(dump_path):
        action = pickle.load(open(dump_path, 'rb'))
    else:
        func_para = {'start_date': start_date, 'end_date': end_date}
        action = part_read_csv(
            action_path, func=filter_date, func_para=func_para)
        del action['model_id']
        # action.fillna(-1, inplace=True)
        # with open(dump_path, 'wb') as dump:
        #     pickle.dump(action, dump)
    return action


def get_actions(start_date, end_date):
    """
    :param start_date:
    :param end_date:
    :return: actions: pd.Dataframe
    """
    dump_path = 'cache/all_action_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        actions = None
        for i in [1, 2, 3]:
            if start_date < '2016-0%d-01' % (i + 2) and end_date > '2016-0%d-01' % (i + 1):
                if actions is None:
                    actions = get_action(i, start_date, end_date)
                else:
                    actions = pd.concat(
                        [actions, get_action(i, start_date, end_date)])
            else:
                continue
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_action_feat(start_date, end_date, base_actions=None):
    dump_path = 'cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        if base_actions is None:
            base_actions = get_actions(start_date, end_date)
        else:
            base_actions = filter_date(base_actions, start_date, end_date)
        actions = base_actions[['user_id', 'sku_id', 'type']]
        df = pd.get_dummies(
            actions['type'], prefix='action')
        actions = pd.concat([actions, df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        del actions['type']
        # pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_action_feat(start_date, end_date, base_actions=None):
    dump_path = 'cache/action_accumulate_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        if base_actions is None:
            base_actions = get_actions(start_date, end_date)
        else:
            base_actions = filter_date(base_actions, start_date, end_date)
        df = pd.get_dummies(base_actions['type'], prefix='action')
        actions = pd.concat([base_actions, df], axis=1)  # type: pd.DataFrame
        # 近期行为按时间衰减
        actions['weights'] = actions['time'].map(lambda x: datetime.strptime(
            end_date, '%Y-%m-%d') - datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        #actions['weights'] = time.strptime(end_date, '%Y-%m-%d') - actions['datetime']
        actions['weights'] = actions['weights'].map(
            lambda x: math.exp(-x.days))
        print(actions.head(10))
        actions['action_1'] = actions['action_1'] * actions['weights']
        actions['action_2'] = actions['action_2'] * actions['weights']
        actions['action_3'] = actions['action_3'] * actions['weights']
        actions['action_4'] = actions['action_4'] * actions['weights']
        actions['action_5'] = actions['action_5'] * actions['weights']
        actions['action_6'] = actions['action_6'] * actions['weights']
        del actions['model_id']
        del actions['type']
        del actions['time']
        del actions['datetime']
        del actions['weights']
        actions = actions.groupby(
            ['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        # pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_user_feat(start_date, end_date, base_actions=None):
    feature = ['user_action_1_ratio', 'user_action_2_ratio', 'user_action_3_ratio',
               'user_action_5_ratio', 'user_action_6_ratio']
    dump_path = 'cache/user_feat_accumulate_%s_%s.pkl' % (
        start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        if base_actions is None:
            base_actions = get_actions(start_date, end_date)
        else:
            base_actions = filter_date(base_actions, start_date, end_date)
        df = pd.get_dummies(base_actions['type'], prefix='action')
        actions = pd.concat([base_actions['user_id'], df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_action_1_ratio'] = actions['action_4'] / \
            actions['action_1']
        actions['user_action_2_ratio'] = actions['action_4'] / \
            actions['action_2']
        actions['user_action_3_ratio'] = actions['action_4'] / \
            actions['action_3']
        actions['user_action_5_ratio'] = actions['action_4'] / \
            actions['action_5']
        actions['user_action_6_ratio'] = actions['action_4'] / \
            actions['action_6']
        for rate_item in feature:
            actions[rate_item].replace(
                np.inf, actions[actions.action_4 > 0][rate_item].quantile(0.5), inplace=True)
        # 同时还产生一些NaN(被除数是0, 除数是0), 采用0来填补这些值
        actions.fillna(0, inplace=True)
        actions = actions[['user_id'] + feature]
        # pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_product_feat(start_date, end_date, base_actions=None):
    feature = ['product_action_1_ratio', 'product_action_2_ratio', 'product_action_3_ratio',
               'product_action_5_ratio', 'product_action_6_ratio']
    dump_path = 'cache/product_feat_accumulate_%s_%s.pkl' % (
        start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        if base_actions is None:
            base_actions = get_actions(start_date, end_date)
        else:
            base_actions = filter_date(base_actions, start_date, end_date)
        df = pd.get_dummies(base_actions['type'], prefix='action')
        actions = pd.concat([base_actions['sku_id'], df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_action_1_ratio'] = actions['action_4'] / \
            actions['action_1']
        actions['product_action_2_ratio'] = actions['action_4'] / \
            actions['action_2']
        actions['product_action_3_ratio'] = actions['action_4'] / \
            actions['action_3']
        actions['product_action_5_ratio'] = actions['action_4'] / \
            actions['action_5']
        actions['product_action_6_ratio'] = actions['action_4'] / \
            actions['action_6']
        # 上面的过程可能产生一些无穷大的值(被除数不是0, 除数是0), 采用中位数来填补这些值
        for rate_item in feature:
            actions[rate_item].replace(
                np.inf, actions[actions.action_4 > 0][rate_item].quantile(0.5), inplace=True)
        # 同时还产生一些NaN(被除数是0, 除数是0), 采用0来填补这些值
        actions.fillna(0, inplace=True)
        actions = actions[['sku_id'] + feature]
        # pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_labels(start_date, end_date):
    dump_path = 'cache/labels_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        base_actions = get_actions(start_date, end_date)
        actions = base_actions[base_actions['type'] == 4]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id', 'label']]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def make_set(start_date, end_date, is_train=True, is_cate8=False):
    if is_train:
        test_start_date = end_date
        test_end_date = date_change(end_date, 5)
        dump_path = 'cache/train_set_%s_%s_%s_%s.pkl' % (
            start_date, end_date, test_start_date, test_end_date)
    else:
        dump_path = 'cache/test_set_%s_%s.pkl' % (start_date, end_date)

    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        user = get_basic_user_feat()
        product = get_basic_product_feat()
        comment_acc = get_comments_product_feat(end_date)
        base_actions = get_actions(start_date, end_date)
        user_acc = get_accumulate_user_feat(start_date, end_date, base_actions)
        product_acc = get_accumulate_product_feat(
            start_date, end_date, base_actions)
        # generate 时间窗口
        # actions = get_accumulate_action_feat(train_start_date, train_end_date)
        actions = None
        for i in (1, 2, 3, 5, 7, 10, 15, 21, 30):
            start = (datetime.strptime(end_date, '%Y-%m-%d') -
                     timedelta(days=i)).strftime('%Y-%m-%d')
            if start < start_date:
                continue
            temp_actions = get_action_feat(start, end_date, base_actions)
            temp_actions.columns = temp_actions.columns.map(
                lambda x: x + '_%s' % i if 'action' in x else x)
            if actions is None:
                actions = temp_actions
            else:
                actions = pd.merge(actions, temp_actions,
                                   how='left', on=['user_id', 'sku_id'])

        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left', on='sku_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')
        if is_train:
            labels = get_labels(test_start_date, test_end_date)
            actions = pd.merge(actions, labels, how='left',
                               on=['user_id', 'sku_id'])
        if is_cate8:
            actions = actions[actions['cate'] == 8]
        actions = actions.fillna(0)
        pickle.dump(actions, open(dump_path, 'wb'))

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    if is_train:
        labels = actions['label'].copy()
        del actions['label']
        return users, actions, labels
    else:
        return users, actions


# def filter_pro(data):
#     drop_list = []
#     pro = part_read_csv(product_path)
#     for each in data.iterrows():
#         if each[1].sku_id not in pro['sku_id']:
#             drop_list.append(each[0])
#     data.drop(drop_list, inplace=True)


def report(pred, fact):
    pro = pickle.load(open('cache/basic_product.pkl', 'rb'))

    pred_buy = pred[pred.label == 1]
    pred_buy['user_id'] = pred_buy['user_id'].astype(int)
    pred_buy['pair'] = pred_buy['user_id'].map(
        str) + '-' + pred_buy['sku_id'].map(str)

    fact_buy = fact[fact.label == 1]
    fact_buy['user_id'] = fact_buy['user_id'].astype(int)
    fact_buy['pair'] = fact_buy['user_id'].map(
        str) + '-' + fact_buy['sku_id'].map(str)

    pred_buyer = pred_buy['user_id'].unique()
    fact_buyer = fact_buy['user_id'].unique()

    pred_pair = pred_buy['pair'].unique()
    fact_pair = fact_buy['pair'].unique()

    pos, neg = 0, 0
    for each in pred_buyer:
        if each in fact_buyer:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / (pos + neg)
    all_user_recall = 1.0 * pos / len(fact_buyer)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for each in pred_pair:
        if each in fact_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / (pos + neg)
    all_item_recall = 1.0 * pos / len(fact_pair)
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / \
        (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / \
        (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print('F12=' + str(F12))
    print('score=' + str(score))
