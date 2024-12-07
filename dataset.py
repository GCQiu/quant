import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import zscore

class SlidingWindowDataset(Dataset):
    def __init__(self, folder_path, window_size, batch_size,stocks_feature,is_training=True):
        self.folder_path = folder_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.data = []
        self.indices = []
        self.stocks_feature = stocks_feature
        self.y = []
        self.min_max_scaler = MinMaxScaler()

        test_list = os.listdir('./dataset_test_v0')
        # 读取文件夹中的所有CSV文件
        if is_training:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    df = pd.read_csv(file_path)
                    if file_name in test_list:
                        #print(len(df),len(df.iloc[:-100]))
                        self.data.append(df.iloc[:-100])
                    else:
                        self.data.append(df)
                    # self.indices.extend(range(len(df) - window_size))
            print('stocks num:{}'.format(len(self.data)))
            # #v0
            # for stock in tqdm.tqdm(self.data,desc="loading data"):
            #     for i in range(len(stock) - window_size + 1):
            #         #处理开盘价
            #         open = stock['open'][i:i + window_size]
            #         self.stocks_feature['open'].append(open)
            #         # 处理收盘价
            #         close = stock['close'][i:i + window_size]
            #         self.stocks_feature['close'].append(close)
            #         #process ema5
            #         ema5 = stock['EMA_5'][i:i + window_size]
            #         self.stocks_feature['ema5'].append(ema5)
            #         #process ema10
            #         ema10 = stock['EMA_10'][i:i + window_size]
            #         self.stocks_feature['ema10'].append(ema10)
            #         #process ema20
            #         ema20 = stock['EMA_20'][i:i + window_size]
            #         self.stocks_feature['ema20'].append(ema20)
            #         # process a_share_capital
            #         a_share_capital = stock['a_share_capital'][i:i + window_size]
            #         self.stocks_feature['a_share_capital'].append(a_share_capital)
            #         #process rsi5
            #         rsi5 = stock['rsi5'][i:i + window_size]
            #         self.stocks_feature['rsi5'].append(rsi5)
            #         #process rsi10
            #         rsi10 = stock['rsi10'][i:i + window_size]
            #         self.stocks_feature['rsi10'].append(rsi10)
            #         #process rsi14
            #         rsi14 = stock['rsi14'][i:i + window_size]
            #         self.stocks_feature['rsi14'].append(rsi14)
            #         #process Return
            #         Return = stock['Return'][i:i + window_size]
            #         self.stocks_feature['Return'].append(Return)
            #         #process ema5_trend
            #         ema5_trend = stock['EMA_5_trend'][i:i + window_size]
            #         self.stocks_feature['EMA_5_trend'].append(ema5_trend)
            #         #process ema10_trend
            #         ema10_trend = stock['EMA_10_trend'][i:i + window_size]
            #         self.stocks_feature['EMA_10_trend'].append(ema10_trend)
            #         #process ema20_trend
            #         ema20_trend = stock['EMA_20_trend'][i:i + window_size]
            #         self.stocks_feature['EMA_20_trend'].append(ema20_trend)
            #         #process pseudo_y
            #         pseudo_y = stock['pseudo_y'][i:i + window_size]
            #         self.stocks_feature['pseudo_y'].append(pseudo_y)
            #         #process volume 越靠近1成交量越大
            #         volume = stock['volume'][i:i + window_size]
            #         self.stocks_feature['volume'].append(volume)
            #         #process turnover_rate
            #         turnover_rate = stock['turnover_rate'][i:i + window_size]
            #         self.stocks_feature['turnover_rate'].append(turnover_rate)
            #         # process turnover
            #         turnover = stock['turnover'][i:i + window_size]
            #         self.stocks_feature['turnover'].append(turnover)
            #         #process type
            #         type = stock['type'][i:i + window_size]
            #         self.stocks_feature['type'].append(type)
            #
            #         label_y = stock['y'][i:i + window_size]
            #         label_y = np.array(label_y)
            #         self.y.append(label_y)
            #v1
            for stock in tqdm.tqdm(self.data,desc="loading data"):
                for feature in self.stocks_feature.keys():
                    for i in range(5,len(stock) - window_size + 1,1):
                        open = stock[feature][i:i + window_size]
                        self.stocks_feature[feature].append(open)
            print(f"num_samples {len(self.stocks_feature['open'])} num_y {len(self.y)}")

            self.num_samples = len(self.stocks_feature['open'])

    def __len__(self):
        return self.num_samples


    def scale_to_range(self,column):
        min_val = np.min(column)
        max_val = np.max(column)
        return (column - min_val) / (max_val - min_val)

    def __getitem__(self, idx):  #v0
        input = []
        # process close
        close = zscore(self.stocks_feature['close'][idx].values)
        input.append(close)
        # process open
        open = zscore(self.stocks_feature['open'][idx].values)
        input.append(open)
        # process next_open
        next_open = zscore(self.stocks_feature['next_open'][idx].values)
        input.append(next_open)
        # process ema5
        EMA_5 = zscore(self.stocks_feature['EMA_5'][idx].values)
        input.append(EMA_5)
        # process ema10
        EMA_10 = zscore(self.stocks_feature['EMA_10'][idx].values)
        input.append(EMA_10)
        # process ema20
        EMA_20 = zscore(self.stocks_feature['EMA_20'][idx].values)
        input.append(EMA_20)
        # process rsi5
        rsi5 = zscore(self.stocks_feature['rsi5'][idx].values)
        input.append(rsi5)
        # process rsi10
        rsi10 = zscore(self.stocks_feature['rsi10'][idx].values)
        input.append(rsi10)
        # process rsi14
        rsi14 = zscore(self.stocks_feature['rsi14'][idx].values)
        input.append(rsi14)
        # process Return
        Return = zscore(self.stocks_feature['Return'][idx].values)
        input.append(Return)
        # process a_share_capital
        a_share_capital_min_max = self.stocks_feature['a_share_capital_percentage'][idx].values
        input.append(a_share_capital_min_max)
        # process float_a_share_capital
        float_a_share_capital_min_max = zscore(
            self.stocks_feature['float_a_share_capital_percentage'][idx].values)
        input.append(float_a_share_capital_min_max)
        # process ema5_trend
        ema5_trend = zscore(self.stocks_feature['EMA_5_trend'][idx].values)
        input.append(ema5_trend)
        # process ema10_trend
        ema10_trend = zscore(self.stocks_feature['EMA_10_trend'][idx].values)
        input.append(ema10_trend)
        # process ema20_trend
        ema20_trend = zscore(self.stocks_feature['EMA_20_trend'][idx].values)
        input.append(ema20_trend)
        # process pseudo_y
        pseudo_y = zscore(self.stocks_feature['pseudo_y'][idx].values)
        input.append(pseudo_y)
        #process volume 越靠近1成交量越大
        volume_rank = self.stocks_feature['volume'][idx].rank(method='min')
        volume_rank_min_max = zscore(volume_rank.values)
        input.append(volume_rank_min_max)
        # process turnover_rate
        turnover_rate_rank = self.stocks_feature['turnover_rate'][idx].rank(method='min')
        turnover_rate_min_max = zscore(turnover_rate_rank.values)
        input.append(turnover_rate_min_max)
        # process turnover
        turnover_rank = self.stocks_feature['turnover'][idx].rank(method='min')
        turnover_rank_min_max = zscore(turnover_rank.values)
        input.append(turnover_rank_min_max)
        # process type
        # type = self.stocks_feature['type'][idx]
        # type = type / 2
        # input.append(type)
        sample = np.array(input)
        label = zscore(self.stocks_feature['y'][idx].values)


        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(0)

    def test_preprocess_v1(self, data):  #v1
        stock_window_feature = []
        # 处理开盘价
        input = stock_window_feature
        # process close
        close = zscore(data['close'].values)
        input.append(close)
        # process open
        open = zscore(data['open'].values)
        input.append(open)
        # process next_open
        next_open = zscore(data['next_open'].values)
        input.append(next_open)
        # process ema5
        EMA_5 = zscore(data['EMA_5'].values)
        input.append(EMA_5)
        # process ema10
        EMA_10 = zscore(data['EMA_10'].values)
        input.append(EMA_10)
        # process ema20
        EMA_20 = zscore(data['EMA_20'].values)
        input.append(EMA_20)
        # process rsi5
        rsi5 = zscore(data['rsi5'].values)
        input.append(rsi5)
        # process rsi10
        rsi10 = zscore(data['rsi10'].values)
        input.append(rsi10)
        # process rsi14
        rsi14 = zscore(data['rsi14'].values)
        input.append(rsi14)
        # process Return
        Return = zscore(data['Return'].values)
        input.append(Return)
        # process a_share_capital
        a_share_capital_min_max = zscore(data['a_share_capital_percentage'].values)
        input.append(a_share_capital_min_max)
        # process float_a_share_capital
        float_a_share_capital_min_max = zscore(
            data['float_a_share_capital_percentage'].values)
        input.append(float_a_share_capital_min_max)
        # process ema5_trend
        ema5_trend = zscore(data['EMA_5_trend'].values)
        input.append(ema5_trend)
        # process ema10_trend
        ema10_trend = zscore(data['EMA_10_trend'].values)
        input.append(ema10_trend)
        # process ema20_trend
        ema20_trend = zscore(data['EMA_20_trend'].values)
        input.append(ema20_trend)
        # process pseudo_y
        pseudo_y = zscore(data['pseudo_y'].values)
        input.append(pseudo_y)
        # process volume 越靠近1成交量越大
        volume_rank = data['volume'].rank(method='min')
        volume_rank_min_max = zscore(volume_rank.values)
        input.append(volume_rank_min_max)
        # process turnover_rate
        turnover_rate_rank = data['turnover_rate'].rank(method='min')
        turnover_rate_min_max = zscore(turnover_rate_rank.values)
        input.append(turnover_rate_min_max)
        # process turnover
        turnover_rank = data['turnover'].rank(method='min')
        turnover_rank_min_max = zscore(turnover_rank.values)
        input.append(turnover_rank_min_max)
        # process type
        # type = self.stocks_feature['type'][idx]
        # type = type / 2
        # input.append(type)

        sample = np.array(input)
        label_y = data['y'].values

        return input, label_y



class SlidingWindowDataset_test(Dataset):
    def __init__(self, folder_path, input_columns, target_column, window_size, batch_size):
        self.folder_path = folder_path
        self.input_columns = input_columns
        self.target_column = target_column
        self.window_size = window_size
        self.batch_size = batch_size
        self.data = []
        self.indices = []
        self.stocks_feature = []
        self.y = []
        self.min_max_scaler = MinMaxScaler()

        # 读取文件夹中的所有CSV文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                self.data.append(df)
                # self.indices.extend(range(len(df) - window_size))
        print('stocks num:{}'.format(len(self.data)))

        #self.stocks_feature, self.y = self.preprocess_df(self.data,self.window_size)

        print(f"num_samples {len(self.stocks_feature)} num_y {len(self.y)}")

        self.num_samples = len(self.stocks_feature)

    def preprocess_df(self,stock,window_size):
        stocks_feature,y= [],[]
        for i in range(len(stock) - self.window_size + 1):
            stock_window_feature = []
            #处理开盘价
            open = stock['open'][i:i + window_size]
            open_min_max = self.min_max_scaler.fit_transform(open.values.reshape(-1, 1))
            open_min_max = open_min_max.flatten()
            stock_window_feature.append(open_min_max)
            # 处理收盘价
            close = stock['close'][i:i + window_size]
            close_min_max = self.min_max_scaler.fit_transform(close.values.reshape(-1, 1))
            close_min_max = close_min_max.flatten()
            stock_window_feature.append(close_min_max)
            #process ema5
            ema5 = stock['EMA_5'][i:i + window_size]
            ema5 = self.min_max_scaler.fit_transform(ema5.values.reshape(-1, 1))
            ema5 = ema5.flatten()
            stock_window_feature.append(ema5)
            #process ema10
            ema10 = stock['EMA_10'][i:i + window_size]
            ema10 = self.min_max_scaler.fit_transform(ema10.values.reshape(-1, 1))
            ema10 = ema10.flatten()
            stock_window_feature.append(ema10)
            #process ema20
            ema20 = stock['EMA_20'][i:i + window_size]
            ema20 = self.min_max_scaler.fit_transform(ema20.values.reshape(-1, 1))
            ema20 = ema20.flatten()
            stock_window_feature.append(ema20)
            # process a_share_capital
            a_share_capital = stock['a_share_capital'][i:i + window_size]
            a_share_capital_min_max = self.min_max_scaler.fit_transform(a_share_capital.values.reshape(-1, 1))
            a_share_capital_min_max = a_share_capital_min_max.flatten()
            stock_window_feature.append(a_share_capital_min_max)
            #process rsi5
            rsi5 = stock['rsi5'][i:i + window_size]/100
            stock_window_feature.append(rsi5)
            #process rsi10
            rsi10 = stock['rsi10'][i:i + window_size]/100
            stock_window_feature.append(rsi10)
            #process rsi14
            rsi14 = stock['rsi14'][i:i + window_size]/100
            stock_window_feature.append(rsi14)
            #process Return
            Return = stock['Return'][i:i + window_size]
            stock_window_feature.append(Return)
            #process ema5_trend
            ema5_trend = stock['EMA_5_trend'][i:i + window_size]
            stock_window_feature.append(ema5_trend)
            #process ema10_trend
            ema10_trend = stock['EMA_10_trend'][i:i + window_size]
            stock_window_feature.append(ema10_trend)
            #process ema20_trend
            ema20_trend = stock['EMA_20_trend'][i:i + window_size]
            stock_window_feature.append(ema20_trend)
            #process pseudo_y
            pseudo_y = stock['pseudo_y'][i:i + window_size]
            stock_window_feature.append(pseudo_y)
            #process volume 越靠近1成交量越大
            volume_rank = stock['volume'][i:i + window_size].rank(method='min')
            volume_rank_min_max = self.min_max_scaler.fit_transform(volume_rank.values.reshape(-1, 1))
            volume_rank_min_max = volume_rank_min_max.flatten()
            stock_window_feature.append(volume_rank_min_max)
            #process turnover_rate
            turnover_rate_rank = stock['turnover_rate'][i:i + window_size].rank(method='min')
            turnover_rate_min_max = self.min_max_scaler.fit_transform(turnover_rate_rank.values.reshape(-1, 1))
            turnover_rate_min_max = turnover_rate_min_max.flatten()
            stock_window_feature.append(turnover_rate_min_max)
            # process turnover
            turnover_rank = stock['turnover'][i:i + window_size].rank(method='min')
            turnover_rank_min_max = self.min_max_scaler.fit_transform(turnover_rank.values.reshape(-1, 1))
            turnover_rank_min_max = turnover_rank_min_max.flatten()
            stock_window_feature.append(turnover_rank_min_max)
            #process type
            type = stock['type'][i:i + window_size]
            type = type/2
            stock_window_feature.append(type)

            stock_window_feature = np.array(stock_window_feature)
            stocks_feature.append(stock_window_feature)

            label_y = stock['y'][i:i + window_size]*10
            label_y = np.array(label_y)
            y.append(label_y)
        #print(stocks_feature)
        return stock_window_feature, label_y

    def __len__(self):
        return self.num_samples // self.batch_size

    def __getitem__(self, idx):

        sample = self.stocks_feature[idx]
        label = self.y[idx]

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# 示例使用
if __name__ == '__main__':
    folder_path = './dataset_useful_case_v0'  # 替换为你的文件夹路径
    input_columns = ['percentage']  # 替换f为你的输入列索引
    target_column = 'y'  # 替换为你的目标列索引
    window_size = 30  # 替换为你想要的窗口大小
    batch_size = 32  # 替换为你想要的batch大小

    dataset = SlidingWindowDataset(folder_path, input_columns, target_column, window_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 遍历数据加载器
    for batch_X, batch_y in dataloader:
        print(batch_X.shape)  # 输出: torch.Size([batch_size, window_size, num_features])
        print(batch_y.shape)  # 输出: torch.Size([batch_size, 1])