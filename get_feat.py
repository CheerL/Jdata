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
begin_date = '2016-02-01'


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
        sex_list_before = [0, 1, 2]
        sex_list_after = [-1, 0, 1]
        user['age'].replace(age_list_before, age_list_after, inplace=True)
        user['sex'].replace(sex_list_before, sex_list_after, inplace=True)
        user.fillna(-1, inplace=True)
        user['age'] = user['age'].astype(int)
        user['sex'] = user['sex'].astype(int)
        del user['user_reg_tm']
        pickle.dump(user, open(dump_path, 'wb'))
    return user


def get_basic_product_feat():
    dump_path = 'cache/basic_product.pkl'
    if os.path.exists(dump_path):
        product = pickle.load(open(dump_path, 'rb'))
    else:
        product = part_read_csv(product_path)
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
        del comments['dt']
        pickle.dump(comments, open(dump_path, 'wb'))
    return comments


def filter_date(action, start_date, end_date):
    return action[(action.time >= start_date) & (action.time < end_date)]


def get_action(i, start_date, end_date):
    action_path = action_base_path + "%d.csv" % (i + 1)
    func_para = {'start_date': start_date, 'end_date': end_date}
    action = part_read_csv(action_path,
                           func=filter_date,
                           func_para=func_para)
    del action['model_id']
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
        # pickle.dump(actions, open(dump_path, 'wb'))
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
        actions = base_actions[['user_id', 'sku_id', 'cate', 'brand', 'type']]
        type_df = pd.get_dummies(actions['type'], prefix='action')
        actions = pd.concat([actions, type_df], axis=1)  # type: pd.DataFrame
        actions = actions.groupby(
            ['user_id', 'sku_id', 'cate', 'brand'], as_index=False).sum()
        del actions['type']
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_user_feat(end_date, start_date=begin_date, base_actions=None):
    feature = ['user_1_ratio', 'user_2_ratio',
               'user_3_ratio', 'user_5_ratio', 'user_6_ratio']
    dump_path = 'cache/user_feat_accumulate_%s_%s.pkl' % (
        start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        if base_actions is None:
            base_actions = get_actions(start_date, end_date)
        else:
            base_actions = filter_date(base_actions, start_date, end_date)
        type_df = pd.get_dummies(base_actions['type'], prefix='action')
        actions = pd.concat([base_actions['user_id'], type_df], axis=1)
        actions = actions.groupby(['user_id'], as_index=False).sum()
        actions['user_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['user_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['user_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['user_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['user_6_ratio'] = actions['action_4'] / actions['action_6']
        # 上面的过程可能产生一些无穷大的值(被除数不是0, 除数是0), 采用中位数来填补这些值
        for rate_item in feature:
            actions[rate_item].replace(np.inf, actions[(actions.action_4 > 0) & (
                actions[rate_item] != np.inf)][rate_item].quantile(0.5), inplace=True)
        # 同时还产生一些NaN(被除数是0, 除数是0), 采用0来填补这些值
        actions.fillna(0, inplace=True)
        actions = actions[['user_id'] + feature]
        # pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def get_accumulate_product_feat(end_date, start_date=begin_date, base_actions=None):
    feature = ['product_1_ratio', 'product_2_ratio',
               'product_3_ratio', 'product_5_ratio', 'product_6_ratio']
    dump_path = 'cache/product_feat_accumulate_%s_%s.pkl' % (
        start_date, end_date)
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb'))
    else:
        if base_actions is None:
            base_actions = get_actions(start_date, end_date)
        else:
            base_actions = filter_date(base_actions, start_date, end_date)
        type_df = pd.get_dummies(base_actions['type'], prefix='action')
        actions = pd.concat([base_actions['sku_id'], type_df], axis=1)
        actions = actions.groupby(['sku_id'], as_index=False).sum()
        actions['product_1_ratio'] = actions['action_4'] / actions['action_1']
        actions['product_2_ratio'] = actions['action_4'] / actions['action_2']
        actions['product_3_ratio'] = actions['action_4'] / actions['action_3']
        actions['product_5_ratio'] = actions['action_4'] / actions['action_5']
        actions['product_6_ratio'] = actions['action_4'] / actions['action_6']
        # 上面的过程可能产生一些无穷大的值(被除数不是0, 除数是0), 采用中位数来填补这些值
        for rate_item in feature:
            actions[rate_item].replace(np.inf, actions[(actions.action_4 > 0) & (
                actions[rate_item] != np.inf)][rate_item].quantile(0.5), inplace=True)
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
        actions = actions[['user_id', 'sku_id', 'cate', 'brand', 'label']]
        pickle.dump(actions, open(dump_path, 'wb'))
    return actions


def filter_pro(data):
    drop_list = []
    pro = part_read_csv(product_path)
    for each in data.iterrows():
        if each[1].sku_id not in pro['sku_id'].values:
            drop_list.append(each[0])
    data.drop(drop_list, inplace=True)


def make_set(end_date, is_train=True, is_cate8=False, is_half=False, is_odd=False):
    date_list = (15, 3, 2, 1)
    start_date = date_change(end_date, -date_list[0])
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
        base_actions = get_actions(begin_date, end_date)
        user_acc = get_accumulate_user_feat(
            end_date, base_actions=base_actions)
        product_acc = get_accumulate_product_feat(
            end_date, base_actions=base_actions)

        actions = None
        base_actions = get_actions(start_date, end_date)
        for i in date_list:
            # for i in (30, 21, 15, 10, 7, 5, 3, 2, 1):
            start = date_change(end_date, -i)
            if start < start_date:
                continue
            temp_actions = get_action_feat(start, end_date, base_actions)
            temp_actions.columns = temp_actions.columns.map(
                lambda x: x + '_%s' % i if 'action' in x else x)
            if actions is None:
                actions = temp_actions
            else:
                actions = pd.merge(actions, temp_actions,
                                   how='outer', on=['user_id', 'sku_id', 'cate', 'brand'])
        if is_train:
            labels = get_labels(test_start_date, test_end_date)
            actions = pd.merge(actions, labels, how='left',
                               on=['user_id', 'sku_id', 'cate', 'brand'])

        actions = pd.merge(actions, user_acc, how='left', on='user_id')
        actions = pd.merge(actions, product_acc, how='left', on='sku_id')
        actions = pd.merge(actions, user, how='left', on='user_id')
        actions = pd.merge(actions, product, how='left',
                           on=['sku_id', 'cate', 'brand'])
        actions = pd.merge(actions, comment_acc, how='left', on='sku_id')

        # 填充缺失值
        actions['age'] = actions['age'].fillna(-1)
        actions['sex'] = actions['sex'].fillna(-1)
        actions['user_lv_cd'] = actions['user_lv_cd'].fillna(1)
        actions['a1'] = actions['a1'].fillna(-1)
        actions['a2'] = actions['a2'].fillna(-1)
        actions['a3'] = actions['a3'].fillna(-1)
        actions['comment_num'] = actions['comment_num'].fillna(0)
        actions.fillna(0, inplace=True)
        pickle.dump(actions, open(dump_path, 'wb'))

    if is_cate8:
        actions = actions[actions.cate == 8]

    if is_half:
        actions = pd.concat([actions[actions.label == 1].iloc[int(is_odd)::2],
                             actions[actions.label == 0].iloc[int(is_odd)::2]])

    users = actions[['user_id', 'sku_id']].copy()
    del actions['user_id']
    del actions['sku_id']
    if is_train:
        labels = actions['label'].copy()
        del actions['label']
        return users, actions, labels
    else:
        return users, actions


def report(pred, fact=None, pred_end_date=None):
    pred_buy = pred[pred.label == 1]
    pred_buy['user_id'] = pred_buy['user_id'].astype(int)
    pred_buy['pair'] = pred_buy['user_id'].map(
        str) + '-' + pred_buy['sku_id'].map(str)

    if pred_end_date:
        fact = None
        fact = get_actions(pred_end_date, date_change(pred_end_date, 5))
        fact = fact[(fact.cate == 8) & (fact.type == 4)]
        filter_pro(fact)
        fact['label'] = 1
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
    return F11, F12, score
