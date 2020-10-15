# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from al_metrics import *
import math

# 坏客户相关的统计

def get_rule_statis(df, r):
    sr = pd.Series({
        'pred_badrate': df[r].mean(),
        'badrate_before': df['y_label'].mean(),
        'badrate_after': df[df[r] == 0]['y_label'].mean(),
        'accuracy_score': accuracy_score(df['y_label'], df[r]),
        'precision_score': precision_score(df['y_label'], df[r]),
        'recall_score': recall_score(df['y_label'], df[r]),
        'f1_score': f1_score(df['y_label'], df[r]),
        'AUC': get_roc_auc_score(df['y_label'], df[r]),
        'KS': get_ks(df['y_label'], df[r])},
        index=['pred_badrate',
               'badrate_before',
               'badrate_after',
               'accuracy_score',
               'precision_score',
               'recall_score',
               'f1_score',
               'AUC',
               'KS'])
    return sr

def grp_statis(df, groupby, label='y_label', triger_date_label='triger_date'):
    """
    :param df: dataframe
    :param groupby: 分组的列名
    :param label: 样本标签列列名
    :param triger_date_label:申请时间的列名
    :return:dataframe
    """
    df = df.groupby(groupby).apply(lambda df: pd.Series({
        'cnt': df.shape[0],
        'bad_cnt': df[label].sum(),
        'bad_rate': df[label].mean(),
        'begin_date': df[triger_date_label].min(),
        'end_date': df[triger_date_label].max()})).reset_index()

    return df


def get_badrate(df_src, groupby, label):

    df = grp_statis(df_src, ['client_batch', groupby],label)
    df_all = grp_statis(df_src, ['client_batch'],label)
    df_all[groupby] = 'all'

    df_badrate = pd.concat([df, df_all], ignore_index=True)
    df_badrate = df_badrate[['client_batch', groupby, 'cnt', 'bad_cnt', 'bad_rate', 'begin_date', 'end_date']]
    df_badrate = df_badrate.sort_values('client_batch')

    return df_badrate


def splitted_type_badrate(df_src, label):
    df_splitted_type_badrate = get_badrate(df_src, 'type', label)
    df_splitted_type_badrate = df_splitted_type_badrate.pivot_table(
        index='client_batch', columns='type',
        values=['cnt', 'bad_cnt', 'bad_rate', 'begin_date', 'end_date'],
        aggfunc=max)

    df_splitted_type_badrate.columns = ['_'.join(reversed(x)) for x in df_splitted_type_badrate.columns]
    df_splitted_type_badrate = df_splitted_type_badrate.reset_index()
    df_splitted_type_badrate['train_test_begin_date'] = df_splitted_type_badrate.apply(
        lambda r: min(r['train_begin_date'], r['test_begin_date']), axis=1)
    df_splitted_type_badrate['train_test_end_date'] = df_splitted_type_badrate.apply(
        lambda r: max(r['train_end_date'], r['test_end_date']), axis=1)

    selected_cols = [
        'client_batch',
        'all_cnt',
        'all_bad_rate',
        'train_cnt',
        'train_bad_rate',
        'test_cnt',
        'test_bad_rate',
        'oot_cnt',
        'oot_bad_rate',
        'train_test_begin_date',
        'train_test_end_date',
        'oot_begin_date',
        'oot_end_date'
    ]

    df_splitted_type_badrate = pd.DataFrame(df_splitted_type_badrate, columns=selected_cols)

    return df_splitted_type_badrate


def get_df_auc_ks(df_src, groupby, scores, label):
    l = []

    for group_name, df in df_src.groupby(['client_batch', groupby]):
        for score in scores:
            df_score = df[df[score].notnull()]
            if df_score.shape[0]==0 :
                continue

            y_true = df_score[label]
            y_pred = df_score[score]

            df_row = pd.DataFrame()
            df_row.loc[1, 'client_batch'] = group_name[0]
            df_row.loc[1, 'feature'] = score
            df_row.loc[1, groupby] = group_name[1]
            df_row.loc[1, 'ks'] = get_ks(y_true, y_pred)
            df_row.loc[1, 'auc'] = get_roc_auc_score(y_true, y_pred)
            df_row.loc[1, 'match_cnt'] = df_score.shape[0]
            df_row.loc[1, 'match_rate'] = df_score.shape[0] / df.shape[0]
            df_row.loc[1, 'bad_cnt'] = y_true.sum()
            df_row.loc[1, 'bad_rate'] = y_true.mean()

            l.append(df_row)

    return pd.concat(l, ignore_index=True)


def get_df_splitted_type_auc_ks(df_src, scores, label):
    df_selected_all = df_src.copy()
    df_selected_all['type'] = 'all'
    df_selected_with_all = pd.concat([df_src, df_selected_all], ignore_index=True)

    df_splitted_type_auc_ks = get_df_auc_ks(df_selected_with_all, groupby='type', scores=scores, label=label)
    df_splitted_type_auc_ks = df_splitted_type_auc_ks.pivot_table(
        index=['client_batch', 'feature'],
        columns='type',
        values=['auc', 'ks', 'match_cnt', 'match_rate', 'bad_cnt', 'bad_rate'])
    df_splitted_type_auc_ks.columns = ['_'.join(reversed(l)) for l in df_splitted_type_auc_ks.columns]
    df_splitted_type_auc_ks = df_splitted_type_auc_ks.reset_index()

    return df_splitted_type_auc_ks


def get_df_monthly_auc_ks(df_src, scores, label):
    df_monthly_auc_ks = get_df_auc_ks(df_src, 'apply_month', scores, label)
    df_monthly_auc_ks = df_monthly_auc_ks.pivot_table(
        index=['client_batch', 'feature'],
        columns='apply_month',
        values=['auc', 'ks'])
    df_monthly_auc_ks.columns.reorder_levels([1, 0])

    return df_monthly_auc_ks


def psi_statis(df_src, splitted_types, scores):
    def bin_psi(x, y):

        if pd.isnull(y) or y == 0 or pd.isnull(x) or x == 0:
            return None
        else:
            return (x - y) * math.log(x / y)

    if 'train' not in splitted_types:
        print('Error: failt to get psi, for train is not in splitted_types')
        return

    bins = list(range(300, 951, 50))
    l = []
    for (client_batch, splitted_type), df_type in df_src.groupby(['client_batch', 'type']):
        for score in scores:
            df_score = df_type[df_type[score].notnull()]
            df = pd.cut(df_score[score].map(to_score), bins=bins, right=False).value_counts().map(
                lambda v: v / df_score.shape[0] if df_score.shape[0] > 0 else np.nan).to_frame('pct')
            df.index.name = 'bin'
            df.index = df.index.astype(str)
            df = df.reset_index()

            df['client_batch'] = client_batch
            df['type'] = splitted_type
            df['feature'] = score

            l.append(df)

    df_psi_detail = pd.concat(l, ignore_index=True).pivot_table(index=['client_batch', 'feature', 'bin'],
                                                                columns='type', values='pct')
    df_psi_detail.columns = [s + '_pct' for s in df_psi_detail.columns.format()]
    df_psi_detail = df_psi_detail.reset_index()

    for splitted_type in filter(lambda s: s != 'train', splitted_types):
        df_psi_detail['train_{}_psi'.format(splitted_type)] = df_psi_detail.apply(
            lambda r: bin_psi(r['train_pct'], r[splitted_type + '_pct']), axis=1)

    psi_col = list(filter(lambda col: '_psi' in col, df_psi_detail.columns.format()))
    df_psi = df_psi_detail.groupby(['client_batch', 'feature']).sum()[psi_col].reset_index()

    df_psi_detail_sum = df_psi_detail.drop(labels='bin', axis=1).groupby(
        ['client_batch', 'feature']).sum().reset_index()
    df_psi_detail_sum['bin'] = '[sum]'

    df_psi_detail = pd.concat([df_psi_detail, df_psi_detail_sum], ignore_index=True).sort_values(
        ['client_batch', 'feature'])
    df_psi_detail = pd.DataFrame(df_psi_detail, columns=['client_batch', 'feature', 'bin',
                                                         'train_pct', 'test_pct', 'oot_pct', 'train_test_psi',
                                                         'train_oot_psi'])

    return df_psi, df_psi_detail

def crosstab_bins(X,y,bins,reverse_=0):
	xbins = pd.cut(X,bins=bins)
	n = y.count()
	bn = y.sum()
	gn = n - bn       
	out = pd.crosstab(np.array(xbins), y)
	for i in [0.0,1.0]:
		if i not in out.keys():
			out[i] = 0
		else:
			pass
	if reverse_!=1:
		out['group%'] = (out[0]+out[1])/n
		out['good_goodall%'] = out[0]/gn
		out['bad_badall%'] = out[1]/bn
		out['bad%']= out[1]/(out[0] + out[1])
		out['cum_bad%'] =out[1].cumsum()/bn   
		out['cum_good%'] = out[0].cumsum()/gn
		out['cum_reject'] = (out[0]+out[1]).cumsum() 
		out['cum_bad'] = out[1].cumsum()
		out['cum_reject%'] =(out[0]+out[1]).cumsum()/n
		out['cum_bad%_reject'] = out['cum_bad']/out['cum_reject']
		out['lift'] = out['cum_bad%']/out['group%'].cumsum()
	else:
		out = out.sort_index(ascending = False)
		out['group%'] = (out[0]+out[1])/n
		out['good_goodall%'] = out[0]/gn
		out['bad_badall%'] = out[1]/bn
		out['bad%']= out[1]/(out[0] + out[1])
		out['cum_bad%'] =out[1].cumsum()/bn   
		out['cum_good%'] = out[0].cumsum()/gn
		out['cum_reject'] = (out[0]+out[1]).cumsum() 
		out['cum_bad'] = out[1].cumsum()
		out['cum_reject%'] =(out[0]+out[1]).cumsum()/n
		out['cum_bad%_reject'] = out['cum_bad']/out['cum_reject']
		out['lift'] = out['cum_bad%']/out['group%'].cumsum()
	return out

def compute_woe_iv_of_one_grouped_var(data,y='cheat',x = 'office_no_dup_mark',delimiter=', '):
    df = data.copy(deep = True)
    df[y] = df[y].astype(int)
    summary = df.groupby(x,as_index = True).agg({y:[np.size,np.sum,lambda x:np.size(x) - np.sum(x)]})
    summary.columns = ['区间数量','区间坏客户数','区间好客户数']
    summary['区间数量占比'] = summary['区间数量'] / summary['区间数量'].sum()
    summary['坏客户占比'] = summary['区间坏客户数'] / summary['区间坏客户数'].sum()
    summary['好客户占比'] = summary['区间好客户数'] / summary['区间好客户数'].sum()
    summary['累计坏客户占比'] = summary['坏客户占比'].cumsum()
    summary['累计好客户占比'] = summary['好客户占比'].cumsum()
    summary['woe'] = np.log(summary['坏客户占比'] / summary['好客户占比'])
    summary['iv'] = summary['woe'] * (summary['坏客户占比'] - summary['好客户占比'])
    summary['binx'] = summary.index
    summary['区间坏客户占比'] = summary['区间坏客户数'] / summary['区间数量']
    summary['iv_sum'] = summary['iv'].sum()
    summary['order'] = summary['binx'].map(lambda m: float(str(m).split(delimiter)[0][1:]))
    summary = summary.sort_values(by=['order']).drop('order',axis=1)
    summary = summary[['binx','区间数量','区间坏客户数','区间好客户数','区间数量占比','坏客户占比','好客户占比','累计坏客户占比','累计好客户占比','区间坏客户占比','woe','iv','iv_sum']]
    summary[['woe','iv','iv_sum']] = summary[['woe','iv','iv_sum']].applymap(lambda x: round(x,3))
    return({'summary':summary,'iv':summary['iv'].sum(),'woe_mapping':summary[['binx','woe',]],'pct_mapping':summary[['binx','区间数量占比']]})

def compute_woe_iv_psi_of_one_grouped_var(data, flag=True, y='cheat',x = 'office_no_dup_mark',delimiter=', '):
    
    def round_(data,cols=['区间数量占比','坏客户占比','好客户占比','累计坏客户占比','累计好客户占比','区间坏客户占比']):
        data[cols] = data[cols].applymap(lambda x: round(x,6))
        return data

    if flag:
        train = data[data['type']=='train']
        test = data[data['type']=='test']
        train1 = compute_woe_iv_of_one_grouped_var(train,y,x,delimiter)['summary']
        test0 = compute_woe_iv_of_one_grouped_var(test,y,x,delimiter)
        test1 = test0['pct_mapping'].rename(columns={'区间数量占比':'test_区间数量占比'})
        test2 = test0['summary']
        result = pd.merge(train1,test1,how='left',on='binx')
        result['psi'] = sum((result['test_区间数量占比']-result['区间数量占比'])*np.log(result['test_区间数量占比']/result['区间数量占比']))
        result['psi'] = result['psi'].map(lambda x: round(x,6))
        result = round_(result)
        test2 = round_(test2)
        return result.drop('test_区间数量占比',axis=1),test2
    else :
        result = compute_woe_iv_of_one_grouped_var(data,y,x,delimiter)['summary']
        result['psi'] = ''
        result = round_(result)
        return result,None

def cut_bin(data, rule, y='y_label', feature='x', binx='binx',woex='woex',delimiter=', '):
    features = rule[feature].drop_duplicates()
    result1 = data.drop(features,axis=1)
    result2 = data.drop(features,axis=1)
    for col in features:
        ll = []
        for tmp in rule[rule[feature]==col][binx]:
            ll = ll + [tmp.split(delimiter)[0][1:]]
            ll = ll+[tmp.split(delimiter)[1][0:-1]]
        ll = [float(i) for i in set(ll)]
        ll.sort()
        lab = []
        inter = []
        for tmp in ll:
            for tm in rule[rule[feature]==col][binx]:
                if float(tm.split(delimiter)[0][1:]) == tmp:
                    lab = lab + [round(rule[(rule[feature]==col) & (rule[binx]==tm)][woex].iat[0],9)]
                    inter = inter + [rule[(rule[feature]==col) & (rule[binx]==tm)][binx].iat[0]]

        result1[col] = pd.cut(data[col],ll,labels=lab)
        result2[col] = pd.cut(data[col],ll,labels=inter)       
    return result1,result2