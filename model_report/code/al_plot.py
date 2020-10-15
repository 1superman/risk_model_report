from al_data_tool import *
from al_metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn import metrics
from matplotlib.font_manager import FontProperties

def plot_score_badrate(df_bins, label, scores,title, output_path):
    #df_bins format: columns=[label]+scores, value of scores is value bin

    df_bins = df_bins.set_index([label], append=True)[scores].stack()
    df_bins.name = 'bin'
    df_bins = df_bins.reset_index()
    df_bins.rename(columns={'level_2': 'score'}, inplace=True)
    df_bins = df_bins[df_bins[label].notnull() & df_bins['bin'].notnull() & df_bins['score'].notnull()]

    df_badrate = df_bins.groupby(['score', 'bin']).mean()[label].reset_index()
    df_badrate.rename(columns={label: 'badrate'}, inplace=True)

    f, ax = plt.subplots(figsize=(8, 3))
    if df_badrate.shape[0] == 0:
        return -1
    sns.pointplot(data=df_badrate, x='bin', y='badrate', hue='score').set_title(title)
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.savefig(output_path, bbox_inches='tight')

    return

def plot_distribution(df, bin_col, label, title, output_path):
    df = df[df[bin_col].notnull() & df[label].notnull()]

    l = []
    for label_value, df_label in df.groupby(label):
        label_total_cnt = df_label.shape[0]
        for bin_value, df_bin in df_label.groupby(bin_col):
            sr = pd.Series()
            sr['score_bin'] = bin_value
            sr['label'] = label_value
            sr['type'] = 'bin_good / total_good' if label_value == 0 else 'bin_bad / total_bad'
            sr['rate'] = df_bin.shape[0] / label_total_cnt

            l.append(sr.to_frame().T)

    df_4_plot = pd.concat(l, ignore_index=True)
    f, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x='score_bin', y='rate', hue='type', data=df_4_plot).set_title(title)
    plt.xticks(rotation=90)

    plt.savefig(output_path, bbox_inches='tight')

    return

def plot_sample_badrate(df,x_col, title, output_path):
    df = df.sort_values(x_col)
    x = df[x_col]
    y1 = df['cnt']
    y2 = df['bad_rate']

    fig = plt.figure(figsize=(8, 4))
    width = 0.8

    ax1 = fig.add_subplot(111)
    p1 = ax1.bar(x, y1, width, color='#054E9F')
    ax1.set_ylabel(u'total cnt')
    ax1.set_xlabel(x_col)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'y')
    ax2.set_ylabel(u'badrate')

    plt.savefig(output_path, bbox_inches='tight')
    return

def plot_lift(y_true: pd.Series, sr_pred_bins: pd.Series, output_path, title='', precision=3, bins=10, asc=True):
	"""
	采用等频分箱计算lift
	:param ascending: 是否按分箱的正序排列
	:return:
	"""
	data = pd.concat([y_true, sr_pred_bins], axis=1)
	data.columns = ['y_true', 'bin']
	b_t = (data['y_true']==1).sum()

	data = data.groupby('bin').apply(lambda df: pd.Series({'cnt': df.shape[0], 'bad_cnt': df['y_true'].sum()}))
    
	data = data.sort_index(ascending=asc)
	data = data.cumsum()
	data['bad_cnt_rate']= data['bad_cnt']/b_t
	data['cnt_rate']= data['cnt']/y_true.shape[0]
	data['lift'] = data['bad_cnt_rate']/data['cnt_rate']
	data = data.reset_index()
	data.columns = ['score_bin','cnt', 'bad_cnt','bad_cnt_rate','cnt_rate', 'lift']

	fig = plt.figure(figsize=(8, 3))
	ax2 = fig.add_subplot(111)
	x = np.arange(data.shape[0])
	width = 0.4
	p1 = ax2.bar(x, data['bad_cnt_rate'].astype(float), width, color='#ffb6b9')
	p2 = ax2.bar(x+0.4, data['cnt_rate'].astype(float), width, color='#9ddcdc')
	for a, b in zip(x, [round(x,2) for x in data['bad_cnt_rate'].astype(float)]):
		plt.text(a, b, b, ha='center', va='bottom', fontsize=10,color ='#3c4245')
	for a, b in zip(x, [round(x,2) for x in data['cnt_rate'].astype(float)]):
		plt.text(a, b, b, ha='center', va='bottom', fontsize=10,color ='#3c4245')
	ax2.set_ylabel(u'用户数', )
	ax2.set_xlabel(u'等频分箱')
	ax2.set_title(title)
	ax2.set_xticks(x)
	ax2.set_ylim(0, 1.5)
	ax2.set_xticklabels(list(data['score_bin']), rotation=80)
	ax1 = ax2.twinx()
	p3 = ax1.plot(x, [round(x,2) for x in data['lift'].astype(float)],'o-',color = '#d65a31')
	ax1.set_ylabel(u'提升度')
	ax1.set_ylim(0, data['lift'].astype(float).max()*1.2)
	for a, b in zip(x, [round(x,2) for x in data['lift'].astype(float)]):
		plt.text(a, b, b, ha='center', va='bottom', fontsize=10,color ='#3c4245')
	plt.legend((p1[0], p2[0],p3[0]), ('坏客户占总坏客户累积比例', '样本数占总样本数累积比例','提升度'),loc = 'upper right')
	plt.xticks(rotation=80)
	plt.savefig(output_path, bbox_inches='tight')
	return

def plot_roc(y_true: pd.Series, y_pred: pd.Series, output_path, title=''):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    lw = 1
    plt.figure(figsize=(4, 4))
    pylab.plot(fpr, tpr, color='darkorange', lw=lw)
    pylab.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.05])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title(title)
    plt.savefig(output_path, bbox_inches='tight')
    return

def plot_ks(y_true: pd.Series, y_pred: pd.Series, output_path, title=''):
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
	plt.figure(figsize=(4, 4))
	x = np.arange(len(thresholds))
	x_show = np.arange(0, len(thresholds), int(len(thresholds) / 10))
	pylab.plot(x, tpr, lw=1)
	pylab.plot(x, fpr, lw=1)
	pylab.plot(x, tpr - fpr, lw=1, linestyle='--')
	pylab.xticks(x_show, [-thresholds[i] for i in x_show], rotation=90)
	pylab.xlabel('score')
	pylab.title(title)
	plt.savefig(output_path, bbox_inches='tight')
	return


def plot_bins(sr_pred_bins:pd.Series, y:pd.Series, output_path, title=''):
	data = pd.concat([sr_pred_bins, y], axis=1)
	data.columns=['bin','y']
	data = data.groupby('bin').apply(lambda df:df['y'].value_counts().to_frame().T
				).fillna(0).apply(lambda df: pd.Series({'bad_cnt':df[1],
										   'good_cnt':df[0],
										   'bad_rate': df[1] / (df[1] + df[0])}),axis=1)
	data = data.reset_index()
	xticks = np.arange(data.shape[0])
	xticklabels=list(data['bin'])
	y1 = list(data['bad_cnt'])
	y2 = list(data['good_cnt'])
	y3 = list(data['bad_rate'])

	font = FontProperties(fname=r"simsun.ttc", size=14)
	width = 0.8
	fig = plt.figure(figsize=(4, 4))
	ax1 = fig.add_subplot(111)
	p1 = ax1.bar(xticks, y1, width, color='#ffb6b9')
	p2 = ax1.bar(xticks, y2, width, bottom=y1, color='#9ddcdc')
	ax1.set_ylabel(u'用户数', fontproperties=font)
	ax1.set_xlabel(u'分箱编号', fontproperties=font)
	ax1.set_title(title, fontproperties=font)
	ax1.set_xticks(xticks)
	ax1.set_xticklabels(xticklabels, rotation=80)
	ax2 = ax1.twinx()
	p3 = ax2.plot(xticks, y3,'o-', color='#d65a31')
	ax2.set_ylabel(u'区间违约率', fontproperties=font)
	ax2.set_ylim(0, max(y3)*1.2)
	plt.legend((p1[0], p2[0],p3[0]), ('坏客户数量', '好客户数量','区间违约率'))

	plt.savefig(output_path, bbox_inches='tight')
	return

