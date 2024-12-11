import numpy as np
import pandas as pd
import tqdm
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import talib
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from scipy.stats import zscore

def date_num(root_path):
    day_num = []
    os.makedirs('./dataset_30',exist_ok=True)
    for csv in os.listdir(root_path):
        df = pd.read_csv(os.path.join(root_path, csv), skiprows=0, sep=',', header=0,index_col=0)
        num = df.shape[0]
        if num>=30:
            shutil.copyfile(os.path.join(root_path,csv),os.path.join('./dataset_30',csv))
        #day_num.append(num)
    # print(len(day_num))
    # plt.hist(np.array(day_num).T, bins=300, facecolor="blue", edgecolor="black", alpha=0.7,density=False)
    # # 显示横轴标签
    # plt.xlabel("区间")
    # # 显示纵轴标签
    # plt.ylabel("频数/频率")
    # # 显示图标题
    # plt.title("频数/频率分布直方图")
    # plt.show()

def percentage_up_down(root_path):
    os.makedirs('./dataset_30',exist_ok=True)
    for csv in tqdm.tqdm(os.listdir(root_path)):

        increase = [0]
        ret_backward = [0]
        ret_forward = [0]
        df = pd.read_csv(os.path.join(root_path,csv), skiprows=0,  header=0,index_col=0)
        stock_close = df['close']
        stock_open = df['open']
        for i in range(1,len(stock_close)):
            per_close = (stock_close[i]-stock_close[i-1])/stock_close[i-1]  #当日涨跌幅
            per_backward = stock_close[i]/stock_close[i-1]-1  #当日和隔日的比例
            increase.append(per_close)
            ret_backward.append(per_backward)
        # for i in range(0,len(stock_close)-1):
        #     per_forward = stock_close[i+1] / stock_close[i] - 1
        #     ret_forward.append(per_forward)
        #ret_forward.append(0)
        # if '104060600072' in csv:
        #     print(df)
        df['percentage_close'] = increase
        df['ret_backward'] = ret_backward
        #df['ret_forward'] = ret_forward
        #df = df.reset_index(inplace=True)

        # if '104060600072' in csv:
        #     print(df)

        df.to_csv("./dataset_30/{}".format(csv))

def map_to_range(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) * 2 - 1
def y_vis(root_path):
    # os.makedirs('./percentage_y_stocks/',exist_ok=True)
    # for csv in tqdm.tqdm(os.listdir(root_path)):
    #     df = pd.read_csv(os.path.join(root_path,csv), skiprows=0)
    #     percentage_close = df['percentage_close']
    #     if len(percentage_close)<=100:
    #         y = df['y']
    #         ret = df['ret_backward']
    #         plt.figure(dpi=500,figsize=(20,10))
    #         plt.plot([i for i in range(len(y))], y, label='曲线1', color='blue')
    #         plt.plot([i for i in range(len(percentage_close))], percentage_close, label='曲线2', color='red')
    #         #plt.plot([i for i in range(len(ret))], ret, label='曲线3', color='green')
    #         #plt.show()
    #         plt.legend()
    #         plt.savefig('./percentage_y_stocks/{}'.format(csv.replace('.csv','.jpg')), dpi=300)
    a_share_capital = []

    for csv in tqdm.tqdm(os.listdir(root_path)):
        df = pd.read_csv(os.path.join(root_path,csv), skiprows=0)
        #a_share_capital.append(df['a_share_capital'].mean())
        #df['y_scaled'] = df['y'].apply(lambda x: map_to_range(x, df['y'].min(), df['y'].max()))
        df['Return'] = df['close'].shift(-1) / df['close'] - 1
        print(df['mfi'].corr(df['y']))
        # nan = df.isnull().any().any()
        #
        # if nan:
        #     nan_locations = df.isna()
        #     print(csv)
        #     # 打印包含 NaN 值的行和列
        #     print(df[nan_locations.any(axis=1)])


        #print(np.corrcoef(df1,df2))
            #plt.show()
    # print(max(a_share_capital),min(a_share_capital))
    # # 生成一个从0到1万亿，以10亿为间隔的整数列表
    # intervals = [0,10,100,1000,10000]
    # intervals = [i * 10**8 for i in intervals]
    # print(intervals)
    #
    # a_share_capital = pd.Series(a_share_capital)
    # binned_series = pd.cut(a_share_capital, bins=intervals)
    # print(binned_series.value_counts())
    # binned_series.value_counts().plot(kind='bar', color='blue')
    # # value_counts = a_share_capital.value_counts()
    # # print(value_counts)


def y_percentage_corre(root_path):
    os.makedirs('./heatmap_stocks/', exist_ok=True)
    for csv in tqdm.tqdm(os.listdir(root_path)):
        df = pd.read_csv(os.path.join(root_path,csv), skiprows=0)
        plt.figure(figsize=(25, 20))
        stock_feature = {
            # # 基本量价因子
            'open':[],
            'close':[],
            'high':[],
            'low':[],
            'next_open':[],
            'volume': [],
            'vwap':[],
            'a_share_capital': [],
            'float_a_share_capital': [],
            'turnover_rate': [],
            'turnover': [],
            # 波动率因子
            'Return': [],
            'EMA_5_trend': [],
            'EMA_10_trend': [],
            'EMA_20_trend': [],
            'pseudo_y': [],
            'low2high': [],
            'klen': [],
            'kup': [],
            'klow': [],
            'ksft': [],
            'klow2':[],
            'ksft2': [],
            'next_open_percentage': [],
            'close_change': [],
            'open_change': [],
            'low_change': [],
            'high_change': [],
            'a_share_capital_percentage': [],
            'float_a_share_capital_percentage': [],
            'vwap_percentage': [],
            'vwap2close':[],
            'volume_change':[],
            'turnover_rate_change':[],
            'turnover_change':[],
            #重叠因子
            'EMA_5': [],
            'EMA_10': [],
            'EMA_20': [],
            'boll_upper':[],
            'boll_middle':[],
            'boll_lower':[],
            'mama':[],
            'fama':[],
            'sar':[],
            'dif':[],
            'dem':[],
            'histogram':[],
            'mom12':[],
            'mom26':[],
            #成交量因子
            'ADOSC':[],
            'obv':[],
            #波动性因子
            'natrPrice_5':[],
            'TRANGE':[],
        'rsi5':[],    'rsi10':[],    'rsi14':[],
            'y':[],
        }
        new_df = pd.DataFrame(df, columns=list(stock_feature.keys()))
                                           # 'volume','vwap','a_share_capital','total_capital','total_capital'\
                                           # ,'float_a_share_capital','turnover','turnover_rate'])


        # 计算DataFrame中所有列之间的相关系数矩阵
        correlation_matrix = new_df.corr()

        # # 使用seaborn的heatmap函数绘制热力图
        # sns.heatmap(correlation_matrix, annot=True, cmap='OrRd',linecolor='black',cbar=True,square=True,)

        # 设置 Seaborn 的样式
        sns.set(style="white")

        # 创建一个掩码，用于隐藏对角线上的值
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # 设置图形大小
        plt.figure(figsize=(20, 15))

        # 绘制热力图
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", annot_kws={"size": 7})

        plt.savefig('./heatmap_stocks/{}'.format(csv.replace('.csv', '.jpg')), dpi=300)


def calculate_base_indicators(root_path, target_path):
    os.makedirs('./{}/'.format(target_path),exist_ok=True)
    os.makedirs('./ema_plot', exist_ok=True)
    for csv in tqdm.tqdm(os.listdir(root_path)):
        data = pd.read_csv(os.path.join(root_path,csv), skiprows=0,index_col=0)
        data = data.copy()

        # data['pseudo_y'] = data['next_open'] / data['open'] - 1
        #
        # data['vwap2close'] = data['vwap']/data['close']
        #
        # data['natrPrice_5'] = talib.NATR(data['high'], data['low'], data['close'], timeperiod=5)
        # data['natrPrice_5'].fillna(method="bfill", inplace=True)
        #
        # data['mfi'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=14)
        # data['mfi'].fillna(method="bfill", inplace=True)
        #
        #
        #
        # data['trangePrice'] = talib.TRANGE(data['high'], data['low'], data['close'])
        # data['trangePrice'].fillna(method="bfill", inplace=True)
        #
        # data['ADOSC']=talib.ADOSC(data['high'],data['low'],data['close'],data['volume'],fastperiod=3,slowperiod=10)
        # data['ADOSC'].fillna(method="bfill", inplace=True)
        #
        # data['obv'] = talib.OBV(data['close'],data['volume'])
        # data['obv'].fillna(method="bfill", inplace=True)
        #
        # data['boll_upper'], data['boll_middle'], data['boll_lower'] = talib.BBANDS(
        #     data['close'],
        #     timeperiod=20,
        #     # number of non-biased standard deviations from the mean
        #     nbdevup=2,
        #     nbdevdn=2,
        #     # Moving average type: simple moving average here
        #     matype=0)
        # data['boll_upper'].fillna(method="bfill", inplace=True)
        # data['boll_middle'].fillna(method="bfill", inplace=True)
        # data['boll_lower'].fillna(method="bfill", inplace=True)
        #
        # mama, fama = talib.MAMA(data['close'], fastlimit=0.5, slowlimit=0.05)
        # data['mama'] = mama
        # data['fama'] = fama
        # data['mama'].fillna(method="bfill", inplace=True)
        # data['fama'].fillna(method="bfill", inplace=True)
        #
        # data['sar'] = talib.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2)
        # data['sar'].fillna(method="bfill", inplace=True)
        # #macd
        # data['dif'], data['dem'], data['histogram'] = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # data['dif'].fillna(method="bfill", inplace=True)
        # data['dem'].fillna(method="bfill", inplace=True)
        # data['histogram'].fillna(method="bfill", inplace=True)
        #
        # data['mom12'] = talib.MOM(data['close'], timeperiod=12)
        # data['mom26'] = talib.MOM(data['close'], timeperiod=26)
        # data['mom12'].fillna(method="bfill", inplace=True)
        # data['mom26'].fillna(method="bfill", inplace=True)
        # data['TRANGE'] = talib.TRANGE(data['high'], data['low'], data['close'])
        # data['TRANGE'].fillna(method="bfill", inplace=True)

        # data['volume_change'] = data['volume']/data['volume'].shift(1)-1
        # data['volume_change'].fillna(method="bfill", inplace=True)
        #
        # data['turnover_rate_change'] = data['turnover_rate'] / data['turnover_rate'].shift(1) - 1
        # data['turnover_rate_change'].fillna(method="bfill", inplace=True)
        #
        # data['turnover_change'] = data['turnover'] / data['turnover'].shift(1) - 1
        # data['turnover_change'].fillna(method="bfill", inplace=True)

        data['ema10_ema5'] = data['EMA_10'] - data['EMA_5']
        data['ema20_ema5'] = data['EMA_20'] - data['EMA_5']

        data['ema5_ema10'] = data['EMA_5'] - data['EMA_10']
        data['ema5_ema20'] = data['EMA_5'] - data['EMA_20']

        data['ema20_ema10'] = data['EMA_20'] - data['EMA_10']
        data['ema10_ema20'] = data['EMA_10'] - data['EMA_20']

        data.to_csv("./{}/{}".format(target_path, csv))






        # data['open_change'] = data['open']/data['open'].shift(1)-1
        # data.loc[0, 'open_change'] = 0
        #
        # data['close_change'] = data['close'] / data['close'].shift(1) - 1
        # data.loc[0, 'close_change'] = 0
        #
        # data['low_change'] = data['low'] / data['low'].shift(1) - 1
        # data.loc[0, 'low_change'] = 0
        #
        # data['high_change'] = data['high'] / data['high'].shift(1) - 1
        # data.loc[0, 'high_change'] = 0
        #
        # data['low2high'] = data['low'] / data['high']
        # data['klen'] = (data['high'] - data['low'])/data['open']
        #
        # data['temp'] = data.apply(lambda row: row['open'] if row['open'] > row['close'] else row['close'], axis=1)
        # data['kup'] = (data['high'] - data['temp'])/data['open']
        # data.drop(['temp'],inplace=True,axis=1)
        #
        # data['temp'] = data.apply(lambda row: row['open'] if row['open'] < row['close'] else row['close'], axis=1)
        # data['klow'] = (data['temp'] - data['low'])/data['open']
        #
        # data['klow2'] = (data['temp'] - data['low'])/(data['high'] - data['low'])
        # data['klow2'] = data['klow2'].fillna(0)
        # data.drop(['temp'], inplace=True,axis=1)
        #
        # data['ksft'] = (2 * data['close'] - data['high'] - data['low'])/data['open']
        #
        # data['ksft2'] = (2 * data['close'] - data['high'] - data['low'])/(data['high'] - data['low'])
        # data['ksft2'] = data['ksft2'].fillna(0)
        #
        # data.to_csv("./{}/{}".format(target_path, csv))

        # # 计算每日涨跌幅
        # data['Return'] = data['close']/data['close'].shift(1)-1
        # data.loc[0, 'Return'] = 0
        #
        # # 计算市值波动
        # data['a_share_capital_percentage'] = data['a_share_capital'] / data['a_share_capital'].shift(1) - 1
        # data.loc[0, 'a_share_capital_percentage'] = 0
        #
        # # 计算流通市值波动
        # data['float_a_share_capital_percentage'] = data['float_a_share_capital'] / data['float_a_share_capital'].shift(1) - 1
        # data.loc[0, 'float_a_share_capital_percentage'] = 0
        #
        # data['vwap_percentage'] = data['vwap'] / data['vwap'].shift(
        #     1) - 1
        # data.loc[0, 'vwap_percentage'] = 0
        #
        # data['next_open_percentage'] = data['next_open'] / data['next_open'].shift(
        #     1) - 1
        # data.loc[0, 'next_open_percentage'] = 0
        #
        # # 计算均线
        # data['EMA_5'] = talib.EMA(data['close'].values, timeperiod=10)
        # data['EMA_10'] = talib.EMA(data['close'].values, timeperiod=15)
        # data['EMA_20'] = talib.EMA(data['close'].values, timeperiod=20)
        #
        # data['EMA_5'].fillna(method="bfill", inplace=True)
        # data['EMA_10'].fillna(method="bfill", inplace=True)
        # data['EMA_20'].fillna(method="bfill", inplace=True)
        #
        # #计算rsi
        # data['rsi5'] = talib.RSI(data['close'], timeperiod=5)
        # data['rsi10'] = talib.RSI(data['close'], timeperiod=10)
        # data['rsi14'] = talib.RSI(data['close'], timeperiod=14)
        #
        # data['rsi5'].fillna(method="bfill", inplace=True)
        # data['rsi10'].fillna(method="bfill", inplace=True)
        # data['rsi14'].fillna(method="bfill", inplace=True)

        #column1_30 = data['EMA_5'].iloc[29:70]

        # 使用MinMaxScaler进行归一化
        #scaler = MinMaxScaler()
        #column1_30_normalized = scaler.fit_transform(column1_30.values.reshape(-1, 1))
        #data['pseudo_y'] = data['close'].shift(-1) / data['close'].shift(-2)

        # 将归一化后的结果转换为DataFrame
        #data['EMA_5_30'] = pd.Series(column1_30_normalized.flatten())

        #计算均线涨跌幅
        # data['EMA_5_trend'] = data['EMA_5']/data['EMA_5'].shift(1)-1
        # data['EMA_5_trend'].fillna(0, inplace=True)
        #
        #
        # data['EMA_10_trend'] = data['EMA_10']/data['EMA_10'].shift(1)-1
        # data['EMA_10_trend'].fillna(0, inplace=True)
        #
        # data['EMA_20_trend'] = data['EMA_20']/data['EMA_20'].shift(1)-1
        # data['EMA_20_trend'].fillna(0, inplace=True)

        # #计算pseudo_y
        # data['pseudo_y'] = data['open']/data['next_open']-1

        #data.to_csv("./{}/{}".format(target_path, csv))

        # plt.figure(dpi=200)
        # plt.plot([i for i in range(len(data['EMA_10']))], data['EMA_10'], label='曲线1', color='blue')

        # fig, axes = plt.subplots(4, 1)
        # axes[0].plot(data['EMA_5'].iloc[29:40], label='Column1')
        # axes[1].plot(data['EMA_5_trend'].iloc[29:40], label='Column2')
        # axes[2].plot(data['close'].iloc[29:40], label='Column2')
        # axes[3].plot(data['a_share_capital'].iloc[29:40], label='Column1')
        #
        # #axes[4].plot(data['pseudo_y'].iloc[29:70], label='Column2')
        # #
        # correlation = data['a_share_capital'].corr(data['close'])
        # print(correlation)


        #plt.savefig('./ema_plot/{}'.format(csv.replace('.csv', '.jpg')))
        # print(data['EMA_5_trend'])



        #print(data['rsi6']/100)
        #print(data['EMA_10'])

        #fig, axes = plt.subplots(3, 1)

        #data[['close', 'EMA_20']].plot(ax=axes[0], grid=True, title='code')
        #data[['rsi6', 'rsi14']].plot(ax=axes[1], grid=True)
        #data[['y']].plot(ax=axes[2], grid=True)
        # plt.figure(dpi=200)
        # plt.plot([i for i in range(len(data['EMA_10']))], data['EMA_10'], label='曲线1', color='blue')
        # plt.plot([i for i in range(len(data['close']))], data['close'], label='曲线2', color='red')

        # fig, axs = plt.subplots(2, 1)
        #
        # # 在第一个子图上绘制正弦波
        # axs[0].plot([i for i in range(len(data['ema']))], data['ema'] ,label='曲线1', color='blue')
        # axs[0].set_title('Sine Wave')
        # axs[0].set_xlabel('x')
        # axs[0].set_ylabel('sin(x)')
        #
        # # 在第二个子图上绘制余弦波
        # axs[1].plot([i for i in range(len(data['close']))], data['close'], label='曲线2', color='red')
        # axs[1].set_xlabel('x')
        #os.makedirs('./dataset_30_indicators/',exist_ok=True)
        #print(data)
        #data.to_csv("./{}/{}".format(target_path,csv))
        #plt.savefig('./ema_plot/{}'.format(csv.replace('.csv', '.jpg')), dpi=300)

#def cal_ema(root_path, window=30):

def calculate_return(root_path):
    #os.makedirs('./ema_plot/',exist_ok=True)
    for csv in tqdm.tqdm(os.listdir(root_path)):
        data = pd.read_csv(os.path.join(root_path,csv), skiprows=0,index_col=0)
        data = data.copy()
        return_1 = []
        return_2 = []
        return_2_1 = []
        close = data['close'].values
        for i in range(len(close)-1):
            daily_return = (close[i+1]/close[i])-1
            return_1.append(daily_return)
        return_1.append(return_1[-1])

        for i in range(len(close)-2):
            daily_return = (close[i+2]/close[i])-1
            return_2.append(daily_return)
        return_2.append(return_2[-1])
        return_2.append(return_2[-1])

        for i in range(len(close)-2):
            daily_return = (close[i+2]/close[i+1])-1
            return_2_1.append(daily_return)
        return_2_1.append(return_2_1[-1])
        return_2_1.append(return_2_1[-1])

        data['return_1'] = return_1
        data['return_2'] = return_2
        data['return_2_1'] = return_2_1

        #bodong率
        print(csv,data)

        #data.to_csv("./dataset_30_indicators/{}".format(csv))

def check_date_continuity(dates, max_gap=10):
    """
    检查日期列表是否连续，最大差值不能超过指定天数。

    参数:
        dates (pd.Series): 日期序列。
        max_gap (int): 最大允许的日期间隔（以天为单位）。

    返回:
        bool: 如果日期连续，则为 True，否则为 False。
    """
    # 计算日期差
    date_diffs = dates.diff().dt.days
    print(list(date_diffs))
    for i in list(date_diffs)[1:]:
        if i >10:
            #print(list(date_diffs).index(i))
            print(dates[list(date_diffs).index(i)-1],dates[list(date_diffs).index(i)],dates[list(date_diffs).index(i)+1])
    # 检查是否存在超过最大间隔的日期差
    return not (date_diffs[1:] > max_gap).any()

def dataset_clean():
    root_path = './dataset_useful_case_v0/'
    #os.makedirs('./dataset',exist_ok=True)
    os.makedirs('./dataset_useful_case_v0_remove_y0',exist_ok=True)
    os.makedirs('./dataset_useful_case_v0_y0',exist_ok=True)
    i=0
    for csv in tqdm.tqdm(os.listdir(root_path)):
        #if '104070300302' in csv:
            df = pd.read_csv(os.path.join(root_path,csv), skiprows=0,index_col=0,dtype={'y': float})
            #print(df['y'])
            zero_count = (df['y'] == 0).sum()
            # non_zero_df = df[df['y'] !=0]
            # zero_df = df[df['y'] == 0]
            # zero_df.reset_index(drop=True, inplace=True)
            #non_zero_df.reset_index(drop=True, inplace=True)
            # if zero_df.empty:
            #     non_zero_df.to_csv("./dataset_30_non_zero/{}".format(csv))
            # else:
            #     non_zero_df.to_csv("./dataset_30_non_zero/{}".format(csv))
            #     zero_df.to_csv("./dataset_30_zero/{}".format(csv))
            if zero_count>0:
                non_zero_df = df[df['y'] != 0]
                non_zero_df.to_csv("./dataset_useful_case_v0_remove_y0/{}".format(csv))

                zero_df = df[df['y'] == 0]
                zero_df.to_csv("./dataset_useful_case_v0_y0/{}".format(csv))
            else:
                shutil.copyfile(os.path.join(root_path, csv), os.path.join('./dataset_useful_case_v0_remove_y0', csv))
            print(len(os.listdir('dataset_useful_case_v0_remove_y0')),len(os.listdir('dataset_useful_case_v0_y0')))

            #     shutil.copyfile(os.path.join(root_path,csv),os.path.join('./dataset_zero',csv))
            # else:
            #     shutil.copyfile(os.path.join(root_path,csv),os.path.join('./dataset_non_zero',csv))

def process_y(root_path):
    for csv in tqdm.tqdm(os.listdir(root_path)):
        df = pd.read_csv(os.path.join(root_path,csv), skiprows=0,index_col=0)
        #df['y_zscore'] = zscore(df['y'])
        #价格
        # scaler = StandardScaler()
        # df['A_zscore'] = scaler.fit_transform(df[['A']])

        # 将 Z-score 归一化到 -1 到 1
        df['y_normalized'] = 2 * (df['y_zscore'] - df['y_zscore'].min()) / (
                    df['y_zscore'].max() - df['y_zscore'].min()) - 1
        df.to_csv("./dataset_train_v0/{}".format(csv))

def calculate_tailb_indicators(root_path):
    for csv in tqdm.tqdm(os.listdir(root_path)):
        df = pd.read_csv(os.path.join(root_path,csv), skiprows=0,index_col=0)
        obv = talib.OBV(df['close'], df['volume']) #
        natrPrice = talib.NATR(df['high'], df['low'], df['close'], timeperiod=10)






def split_train_test(root_path):
    os.makedirs('./dataset_train_v1',exist_ok=True)
    os.makedirs('./dataset_test_v1',exist_ok=True)
    for csv in tqdm.tqdm(os.listdir(root_path)):
        df = pd.read_csv(os.path.join(root_path,csv), skiprows=0,index_col=0)
        day_num = df.shape[0]
        #print(day_num)
        if 100/day_num <=0.3:
            #last_100_rows = df.tail(100)
            remaining_rows = df.iloc[:-100]
            df.to_csv("./dataset_test_v1/{}".format(csv))
            remaining_rows.to_csv("./dataset_train_v1/{}".format(csv))
        else:
            df.to_csv("./dataset_train_v1/{}".format(csv))
    print(len(os.listdir('./dataset_train_v1')),len(os.listdir('dataset_test_v1')))
    # intervals = [0, 10, 100, 1000, 10000]
    # intervals = [i * 10 ** 8 for i in intervals]
    # a_share_capital = []
    # for stock in case:
    #     a_share_capital.append(stock['a_share_capital'].mean())
    # a_share_capital = pd.Series(a_share_capital)
    # binned_series = pd.cut(a_share_capital, bins=intervals)
    # print(binned_series.value_counts())
    # binned_series.value_counts().plot(kind='bar', color='blue')
    # plt.show()
    # print(len(case),len(os.listdir(root_path)))
            #shutil.copyfile(os.path.join(root_path, csv), os.path.join('./dataset_30', csv))








if __name__ == '__main__':
    root_path = './dataset'
    os.makedirs('./dataset_abandon',exist_ok=True)
    os.makedirs('./dataset_out_of_domain',exist_ok=True)
    os.makedirs('./dataset_useful_case',exist_ok=True)

    #process_y('./dataset_test_v0')
    # for csv in tqdm.tqdm(os.listdir('./dataset_non_zero')):
    #     df = pd.read_csv(os.path.join(root_path,csv), skiprows=0,index_col=0)
    #     day_num = df.shape[0]
    #     if day_num<30:
    #         shutil.move(os.path.join('./dataset_non_zero',csv),os.path.join('./dataset_abandon',csv))
    # print(len(os.listdir('./dataset_abandon')),len(os.listdir('./dataset_non_zero')))
    #date_num('./dataset')
    #percentage_up_down(root_path)
    #y_vis('./dataset_train_v0')
    #process_y('./dataset_test_v0')
    #y_percentage_corre('./dataset_train_v0')
    calculate_base_indicators(root_path='./dataset_test_v0',target_path = './dataset_test_v0')
    #calculate_return(root_path)
    #dataset_clean()
    #print(len(os.listdir('./dataset_non_zero')),len(os.listdir('./dataset_zero')),len(os.listdir('./dataset')))
    # print(len(os.listdir('./dataset_temp')))
    #split_train_test('./dataset_useful_case_v0_remove_y0')
    #y_vis('./dataset_test_v0')
    # os.makedirs('./dataset_test_v0_temp', exist_ok=True)
    # for csv in tqdm.tqdm(os.listdir('./dataset_test_v0')):
    #     df1 = pd.read_csv(os.path.join('./dataset_test_v0',csv),index_col=0)
    #     df2 = pd.read_csv(os.path.join('./dataset_train_v0',csv),index_col=0)
    #
    # # 拼接两个 DataFrame
    #     merged_df = pd.concat([df2, df1], ignore_index=False)
    #
    # # 保存拼接后的 DataFrame 到新的 CSV 文件
    #
    #     merged_df.to_csv('./dataset_test_v0_temp/{}'.format(csv))

    # path = 'daily_data.csv'  # 数据存放路径
    # df = pd.read_csv(path, skiprows=0, parse_dates=['date'])
    # stocks_id = list(df['instrument_id'].unique())
    # os.makedirs('./dataset', exist_ok=True)
    # print(len(stocks_id))
    # for id in tqdm.tqdm(stocks_id):
    #     stock = {}
    #     stock = df.loc[(df['instrument_id'] == id)]
    #     # print(stock.reset_index(drop=True))
    #     stock.reset_index(drop=True, inplace=True)
    #     stock.to_csv("./dataset/{}.csv".format(str(id)))
    # print(len(os.listdir('./dataset')))





