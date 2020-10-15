# -*- coding: utf-8 -*-

"""
提供各种数据处理的函数
"""


import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from al_metrics import *
import os
import random
from sklearn.model_selection  import train_test_split
import warnings
warnings.filterwarnings('ignore')

#读取id文件、特征文件、分数文件，并合并,可选择是否进行特征筛选
def get_df_src(cbid, path_df_id,
               feature_dir=None, features=[],
               score_dir=None, scores='all', single_scores_only=False, combine_scores_only=False):

    df_id = pd.read_pickle(path_df_id)
    df_id = df_id.set_index('uid')

    df_feature = pd.DataFrame(index=df_id.index)
    if feature_dir is not None and len(features)>0:
        for f in features:
            df = pd.read_csv(feature_dir + '{}_{}_features.csv'.format(cbid, f))
            df = df.set_index('uid')

            df_feature[df.columns] = df

    df_scores = pd.DataFrame(index=df_id.index)
    if score_dir is not None and len(scores)>0:
        for file in filter(lambda s: '.pkl' in s , os.listdir(score_dir)):
            df = pd.read_pickle(score_dir+file)
            df = df.set_index('uid')
            print(df.columns)
            df_scores[df.columns] = df

        if scores=='all':
            scores=df_scores.columns.format()

        if single_scores_only:
            scores = list(filter(lambda s: '+' not in s,scores))
        elif combine_scores_only:
            scores = list(filter(lambda s: '+' in s, scores))

        df_scores = df_scores[scores]

    df_src = df_id.copy()
    df_src[df_feature.columns] = df_feature
    df_src[df_scores.columns] = df_scores

    return df_src, df_feature.columns.format(), df_scores.columns.format()


import pandas as pd
import numpy as np
import hashlib


def to_md5(s, encoding='utf-8'):
    md = hashlib.md5()
    md.update(str(s).encode(encoding))
    res_md5 = md.hexdigest()
    return res_md5

def md5_uid_df(df):
    df["triger_date"] = df["triger_date"].astype(np.str).apply(lambda x: str(x)[:10])
    new_df = df[["realname", "certid", "mobile", "card", "triger_date"]]
    new_col = pd.DataFrame()
    new_col["one"] = new_df.apply(
        lambda row: str(row['realname']).replace(' ', '') + str(row['certid']).replace(' ', '').upper() + str(
            row['mobile']).replace(' ', '') + str(row['card']).replace(' ', '') + str(row['triger_date']).replace(
            ' ','')[:10], axis=1)
    df['uid'] = new_col["one"].apply(lambda row: to_md5(str(row)))
    return df

def al_train_test_split(df_src, label='y_label', test_size=0.3):
    """
    样本切分函数.先按label分类，每类单独切成train/test，再按train/test合并，
    使得train/test的badrate能高度一致
    :param df_src:
    :param label:
    :param test_size:
    :return:
    """

    l = [[], [], [], []]
    for label_value, X in df_src.groupby(label):

        X[label] = label_value

        row = train_test_split(X.drop(labels=label, axis=1), X[label], test_size=test_size)

        for i in range(0, 4):
            l[i].append(row[i])

    list_df = []
    for i in range(0, 4):
        list_df.append(pd.concat(l[i]))

    return tuple(list_df)

def get_splitted_data(df_selected, label, selected_features):
    X = {}
    y = {}

    X['all'] = df_selected[selected_features]
    y['all'] = df_selected[label]

    for name, df in df_selected.groupby('type'):
        X[name] = df[selected_features]
        y[name] = df[label]

    if not X.__contains__('oot'):
        X['oot'] = None
        y['oot'] = None

    return X['all'], y['all'], X['train'], y['train'], X['test'], y['test'], X['oot'], y['oot']

def filter_features(df_feature_score, df_corr, df_coverage,min_feature_score=1, max_corr=0.9,min_coverage=0.3):
    # 特征筛选：1.特征重要性高于min_feature_score的；
    #         2.特征相关性低于max_corr的（相关性高的，保留特征重要性高的那个）
    
    df_coverage = df_coverage[df_coverage>min_coverage]
    df_feature_score=df_feature_score.loc[df_coverage.index.format()]
    df_feature_score = df_feature_score[df_feature_score['Value'] >= min_feature_score]
    df_feature_score = df_feature_score.sort_values(by='Value', ascending=False)

    selected_features = []
    for idx in df_feature_score.index.format():
        high_corr_cnt = df_corr.loc[idx, selected_features].map(lambda v: 1 if abs(v) >= max_corr else 0).sum()
        if high_corr_cnt == 0:
            selected_features.append(idx)

    df_selected_features = df_feature_score.loc[selected_features].reset_index(
        ).rename(columns={'Type':'feature','Value': 'feature_score'})

    return df_selected_features


def tf_df_fy(df, badrate=0.1):
    """
    根据badrate随机生成y
    :param df:
    :param badrate:
    :return:
    """
    row_cnt = df.shape[0]
    sr = pd.Series(np.random.rand(row_cnt)).map(lambda v: 1 if v <= badrate else 0)
    return sr


def tf_ey(y):
    """
    y加密
    :param y:
    :return:
    """
    def get_odd_map(idx, v, l0):
        if idx == 0:
            l = list(filter(lambda x: x != 0, l0))  # 首位不为0
        else:
            l = l0

        l = list(filter(lambda x: (x + v) % 2 == 0, l))
        random.shuffle(l)
        return l[0]

    if pd.isnull(y) or y not in [0, 1, 0.0, 1.0]:
        return np.nan

    ids = [1, 2]  # odd check id
    map_tuple = (
        (0, [0, 2, 4, 6, 9]),  # y=1 map
        (1, [1, 3, 6, 7, 8])
    )

    v = random.randint(100, 999)
    digits = list(map(lambda s: int(s), str(v)))
    is_odd = list(map(lambda v: v % 2, digits))
    checksum = sum([is_odd[x] for x in ids]) % 2

    if y == 1:
        l = map_tuple[checksum][1]
    else:
        l = list(filter(lambda x: x not in map_tuple[checksum][1], range(0, 10)))

    digits[map_tuple[checksum][0]] = get_odd_map(
        map_tuple[checksum][0],
        digits[map_tuple[checksum][0]],
        l)

    return int(''.join(list(map(lambda v: str(v), digits))))


def tf_dy(v):
    """
    y解密
    :param v:
    :return:
    """
    if pd.isnull(v):
        return np.nan

    ids = [1, 2]  # odd check id
    map_tuple = (
        (0, [0, 2, 4, 6, 9]),  # y=1 map
        (1, [1, 3, 6, 7, 8])
    )

    digits = list(map(lambda s: int(s), str(v)))
    is_odd = list(map(lambda v: v % 2, digits))

    checksum = sum([is_odd[x] for x in ids]) % 2

    if digits[map_tuple[checksum][0]] in map_tuple[checksum][1]:
        return 1
    else:
        return 0


def get_sr_diff_days(sr1, sr2, sr_name):
    return pd.Series((sr1 - sr2).map(lambda v: v.days), name=sr_name)


def datetime_to_days_parallel(df, base_datetime_col, datetime_cols, datetime_columns_transform_type, n_jobs=1):
    """
    日期转天数差
    :param df:
    :param base_datetime_col:
    :param datetime_cols:
    :param datetime_columns_transform_type: should be 'base_datetime-datetime_columns' or 'datetime_columns-base_datetime_time'
    :param n_jobs:
    :return:
    """

    if datetime_columns_transform_type not in ['base_datetime-datetime_columns', 'datetime_columns-base_datetime']:
        raise Exception(
            "datetime_columns_transform_type should be 'base_datetime-datetime_columns' or 'datetime_columns-base_datetime'.")
        return

    df = df.copy()

    l1 = Parallel(n_jobs=n_jobs)(delayed(pd.to_datetime)(df[col]) for col in [base_datetime_col] + datetime_cols)
    df[[base_datetime_col] + datetime_cols] = pd.concat(l1, axis=1)[[base_datetime_col] + datetime_cols]

    l2 = []
    if datetime_columns_transform_type == 'base_datetime-datetime_columns':
        l2 = Parallel(n_jobs=n_jobs)(
            delayed(get_sr_diff_days)(df[base_datetime_col], df[col], col) for col in datetime_cols)

    elif datetime_columns_transform_type == 'datetime_columns-base_datetime':
        l2 = Parallel(n_jobs=n_jobs)(
            delayed(get_sr_diff_days)(df[col], df[base_datetime_col], col) for col in datetime_cols)

    df[datetime_cols] = pd.concat(l2, axis=1)[datetime_cols]

    return df


# 批量等距分箱
def cut_batch(df_src, X, bins=10, n_jobs=1):
    df = df_src.copy()
    data = Parallel(n_jobs=n_jobs)(delayed(pd.cut)(df[col], bins) for col in X if df[col].count() > 0)
    data = pd.concat(data, axis=1)
    for c in data.columns:
        data[c] = data[c].astype(str)

    df.drop(data.columns, axis=1, inplace=True)
    df = pd.concat([df, data], axis=1)
    return df


# 批量等频分箱
def qcut_batch(df_src, X, bins=10, n_jobs=1):
    df = df_src.copy()
    data = Parallel(n_jobs=n_jobs)(delayed(pd.qcut)(df[col], q=bins, duplicates='drop') for col in X if df[col].count() > 0)
    data = pd.concat(data, axis=1)
    for c in data.columns:
        data[c] = data[c].astype(str)

    df.drop(data.columns, axis=1, inplace=True)
    df = pd.concat([df, data], axis=1)
    return df


# 定义一个卡方分箱（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
def chi_merge_cut(df, variable, flag, confidenceVal=3.841, bin=10, sample_rate=0.03, sample=None):
    """
    :param df: 传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    :param variable: 需要卡方分箱的变量名称（字符串）
    :param flag: 正负样本标识的名称（字符串）
    :param confidenceVal: 置信度水平（默认是不进行抽样95%）
    :param bin: 最多箱的数目
    :param sample_rate: 若某一组里的样本数量比例小于该值，进行合并
    :param sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    :return: list
    """
    # 进行是否抽样操作
    if sample is not None:
        df = df.sample(n=sample)
    else:
        df

        # 进行数据格式化录入
    total_num = df.groupby([variable])[flag].count()  # 统计需分箱变量每个值数目
    total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
    positive_class = df.groupby([variable])[flag].sum()  # 统计需分箱变量每个值正样本数
    positive_class = pd.DataFrame({'positive_class': positive_class})  # 创建一个数据框保存之前的结果
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,
                       how='inner')  # 组合total_num与positive_class
    regroup.reset_index(inplace=True)
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数
    regroup = regroup.drop('total_num', axis=1)
    if regroup.shape[0] <= bin:
        if df[variable].isna().sum() > 0:
            regroup = regroup.append({variable: 'null', 'positive_class': df[variable].isna().sum(),
                                      'negative_class': df[variable].isna().shape[0] - df[variable].isna().sum()},
                                     ignore_index=True)
        regroup['variable'] = variable
        regroup.rename(columns={variable: 'interval', 'positive_class': 'flag_1', 'negative_class': 'flag_0'},
                       inplace=True)
        return regroup['interval'].tolist(), regroup
    np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
    # print('已完成数据读入,正在计算数据初处理')

    # 处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (
                np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    # 对相邻两个区间进行卡方值计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
              * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
              ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
                      np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi)
    # print('已完成数据初处理，正在进行卡方分箱核心操作')

    # 把卡方值最小的两个区间进行合并（卡方分箱核心）
    while (1):
        if (len(chi_table) <= (bin - 1) and min(chi_table) >= confidenceVal):
            break
        chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值试最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                            np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                                       np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                        np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                    chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[
                                               chi_min_index, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)

        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                            np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                                       np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                        np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                    chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[
                                               chi_min_index, 2]))
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[
                chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                       * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[
                chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                       ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                   np_regroup[chi_min_index + 1, 1] + np_regroup[
                                               chi_min_index + 1, 2]) * (
                                                    np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (
                                                    np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    # 处理若箱内数量过小的，则进行合并
    while (1):
        if np.size(list(set(np_regroup[:, 1] + np_regroup[:, 2] > df.shape[0] * sample_rate))) == 1:
            break
        i = np.argmin(np_regroup[:, 1] + np_regroup[:, 2])
        if i == 0:
            chi_min_index = 0
        elif i == np_regroup.shape[0] - 1:
            chi_min_index = i - 1
        else:
            chi_min_index = np.argwhere(chi_table == min(chi_table[i - 1], chi_table[i]))[0]
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值试最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                            np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                                       np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) \
                                            * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (
                                                        np_regroup[chi_min_index - 1, 2] + np_regroup[
                                                    chi_min_index, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)

        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                            np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                                       np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) \
                                            * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index, 1]) * (
                                                        np_regroup[chi_min_index - 1, 2] + np_regroup[
                                                    chi_min_index, 2]))
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[
                chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                       * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[
                chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                       ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                   np_regroup[chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) \
                                        * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (
                                                    np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    # print("已完成核心分箱操作，正在保存结果")
    # 把结果保存成一个DataFrame
    result_data = pd.DataFrame()
    result_data['variable'] = [variable] * np_regroup.shape[0]  # 结果表第一列，变量名
    list_temp = []
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:
            x = '(' + str(df[variable].min() - 0.1) + ',' + str(np_regroup[i, 0]) + ')'
        elif i == np_regroup.shape[0] - 1:
            x = '(' + str(np_regroup[i - 1, 0]) + ',' + str(df[variable].max() + 0.1) + ')'
        else:
            x = '(' + str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0]) + ')'
        list_temp.append(x)
    result_data['interval'] = list_temp
    result_data['flag_0'] = np_regroup[:, 2]
    result_data['flag_1'] = np_regroup[:, 1]
    bins = [round(df[variable].min() - 0.1, 3), round(df[variable].max() + 0.1, 3)]
    bins.extend([np_regroup[i, 0] for i in np.arange(np_regroup.shape[0] - 1)])
    bins.sort()
    return bins


# 批量的卡方分箱
def chi_merge_cut_batch(df, X, flag, confidenceVal=3.841, bin=10, sample_rate=0.03, sample=None, n_jobs=1):
    """
    :param df: 传入一个数据集包含需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    :param X: 需要卡方分箱的变量（数组）
    :param flag: 正负样本标识的名称（字符串）
    :param confidenceVal: 置信度水平（默认是不进行抽样95%）
    :param bin: 最多箱的数目
    :param sample_rate: 若某一组里的样本数量比例小于该值，进行合并
    :param sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    :param n_jobs: 最大的并行任务数
    :return:dataframe
    """
    bins_list = Parallel(n_jobs=n_jobs)(delayed(chi_merge_cut)(df, col, flag, confidenceVal, bin, sample_rate, sample)
                                        for col in X if df[col].count() > 0)
    result_list = Parallel(n_jobs=n_jobs)(delayed(pd.cut)(df[col], bins) for col, bins in zip(X, bins_list))
    result_data = pd.concat(result_list, axis=1)
    for c in result_data.columns:
        result_data[c] = result_data[c].astype(str)

    df.drop(result_data.columns, axis=1, inplace=True)
    df = pd.concat([df, result_data], axis=1)

    return df

def mono_cut(x, y, min_bin_pct, min_woe_diff, init_bin_cnt, precision,type='mono'):
    """
    自动单调/单峰分箱。步骤如下：
    1.对x进行初始化qcut，生成df_init_bins
    2.对df_init_bins进行迭代合并，寻找最佳合并方案，用best_merge[i]表示df_init_bins[:i+1]的最佳合并方案。
    其中best_merge[i]=best([
        merge[0:i+1],
        best_merge[0] + merge[1:i+1],
        best_merge[1] + merge[2:i+1],
        ...
        best_merge[i-1] + merge[i:i+1],
        ])
    其中merge[a:b]表示将df_init_bins[a:b]合成一个分箱。
    最佳方案为满足指定条件下（is_ok()）,IV值最大。
    :param x: x series
    :param y: y series
    :param min_bin_pct:
    :param min_woe_diff:
    :param init_bin_cnt:
    :param precision: precision of pd.cut
    :param type: 可选择:
        'mono':单调分箱
        'unimodal':单峰分箱
    :return:
        result_data：分箱后的数据
        result_bins：分箱的边界
    """

    def is_monotonic(l: pd.Series):
        return l.is_monotonic_increasing or l.is_monotonic_decreasing

    def is_unimodal(l):
        l = list(zip(l[:-1], l[1:]))
        l = [x[1] - x[0] for x in l]
        l = list(filter(lambda x: x != 0, l))
        l = list(zip(l[:-1], l[1:]))
        l = [x[1] * x[0] for x in l]
        l = list(filter(lambda x: x < 0, l))
        if len(l) <= 1:
            return True
        else:
            return False

    def is_diff_enough(l, min_diff):
        l = list(zip(l[:-1], l[1:]))
        l = [abs(x[1] - x[0]) > min_diff for x in l]

        return sum(l) == len(l)

    def is_ok(df, min_bin_pct, min_woe_diff, type):
        if (df['pct'] < min_bin_pct).any():
            return False

        if not is_diff_enough(df['woe'], min_woe_diff):
            return False

        if type == 'mono':
            if not is_monotonic(df['woe']):
                return False
        elif type == 'unimodal':
            if not is_unimodal(df['woe']):
                return False
        else:
            raise Exception("unexcepted value of type, should be 'mono' or 'unimodal'")

        return True

    if x.count() / len(x) <= min_bin_pct:
        result_bins=[float('-inf'),float('inf')]
        result_data = pd.cut(x, bins=result_bins, precision=precision).astype(str)

        return result_data, result_bins


    df_data=pd.DataFrame({'x':x, 'y':y})

    _,retbins = pd.qcut(df_data['x'], q=init_bin_cnt,precision=precision,
                        duplicates='drop',retbins=True)
    retbins = np.unique(np.round(retbins,precision))
    retbins[0] = float('-inf')
    retbins[-1] = float('inf')
    df_data['x'] = pd.cut(df_data['x'],bins=retbins,precision=precision)

    df_data.dropna(subset=['x'], inplace=True)
    df_data['x'] = df_data['x'].astype(str)
    df_data['bin_right'] = df_data['x'].map(lambda s: float(''.join(s.split(',')[1][:-1])))#分箱右边界

    df_init_bins = df_data.groupby('bin_right').apply(
        lambda v: pd.Series(v['y'].value_counts().to_dict(),index=[0,1])
        ).fillna(0).rename(columns={0:'good_cnt',1:'bad_cnt'}).sort_index().reset_index()

    total_good_cnt = y.value_counts()[0]
    total_bad_cnt = y.value_counts()[1]
    total_cnt = len(y)


    best_merge=[]# best_merge[i]表示0-(i-1)箱的最佳合并方案，以及相关指标
    l0=[]
    for i in range(df_init_bins.shape[0]):
        l1=[]
        for j in range(i+1):
            #   best_merge[j-1] + merge[j:i+1]
            df = df_init_bins.iloc[j:i+1][['good_cnt','bad_cnt']].sum()
            df['pct'] = (df['bad_cnt'] + df['good_cnt']) / total_cnt
            bin_bad_cnt = df['bad_cnt'] if df['bad_cnt']>0 else df['good_cnt']*0.1
            bin_good_cnt =  df['good_cnt'] if  df['good_cnt']>0 else df['bad_cnt']*0.1
            df['woe'] = math.log((bin_bad_cnt/total_bad_cnt)*(total_good_cnt/bin_good_cnt))
            df['iv'] = (bin_bad_cnt/total_bad_cnt - bin_good_cnt/total_good_cnt)*df['woe']
            df['right']=i

            df = df.to_frame().T

            if j>0:
                df = best_merge[j-1].append(df)

            l1.append(df)

        l0.append(l1)
        l1.sort(key = lambda v: (is_ok(v,min_bin_pct,min_woe_diff,type),v['iv'].sum()),
                reverse=True)
        best_merge.append(l1[0])  #best_merge[i]

    result_bins = [df_init_bins.iloc[i]['bin_right'] for i in best_merge[-1]['right'].astype(int)]
    result_bins = [float('-inf')] + result_bins
    result_bins[-1] = float('inf')
    result_bins = np.round(result_bins,precision)

    result_data = pd.cut(x, bins=result_bins,precision=precision).astype(str)

    return result_data,result_bins

def mono_cut_batch(X, y, min_bin_pct, min_woe_diff, init_bin_cnt, precision,type='mono', n_jobs=1):
    """
    批量单调/单峰分箱，具体用法见mono_cut()
    :param X:
    :param y:
    :param min_bin_pct:
    :param min_woe_diff:
    :param init_bin_cnt:
    :param precision:
    :param type:
    :param n_jobs:
    :return:
        df_to_bins:分箱后的数据
        dict_bins:各特征的分箱方式
    """
    l = Parallel(n_jobs=n_jobs)(delayed(mono_cut)(
        X[c], y, min_bin_pct, min_woe_diff,init_bin_cnt,precision,type) for c in X.columns)


    df_to_bins = pd.concat([v[0] for v in l],axis=1)
    dict_bins = dict([(v[0].name,v[1]) for v in l])
    return df_to_bins,dict_bins

