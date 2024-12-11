import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from dataset import SlidingWindowDataset,SlidingWindowDataset_test
from models.transformer import TransformerModel,TransformerModel_NAT,TransformerModel_reg,TransformerModel_reg_T2V
import numpy as np
import logging
from scipy.stats import zscore
from losses import weighted_mse_loss
import matplotlib.pyplot as plt
from utils import val_NAT
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    stock_feature_all = {
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
            # #成交量因子
            # # 'ADOSC':[],
            # # 'obv':[],
            #波动性因子
            'natrPrice_5':[],
            'TRANGE':[],
        'rsi5':[],    'rsi10':[],    'rsi14':[],
            'y':[],
        }
    stock_feature_v0 = {
        # 基本量价因子
        'open': [],
        'close': [],
        'high': [],
        'low': [],
        'next_open': [],
        'volume': [],
        'vwap': [],
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
        'next_open_percentage': [],
        'close_change': [],
        'open_change': [],
        'low_change': [],
        'high_change': [],
        'a_share_capital_percentage': [],
        'float_a_share_capital_percentage': [],
        'vwap_percentage': [],
        'vwap2close': [],
        'volume_change': [],
        'turnover_rate_change': [],
        'turnover_change': [],
        # # 重叠因子
        'EMA_5': [],
        'EMA_10': [],
        'EMA_20': [],
        'boll_upper': [],
        'boll_middle': [],
        'boll_lower': [],
        'mama': [],
        'fama': [],
        'sar': [],
        'dif': [],
        'dem': [],
        'histogram': [],
        'mom12': [],
        'mom26': [],
        # # #成交量因子
        # # # 'ADOSC':[],
        # # # 'obv':[],
        # # 波动性因子
        'natrPrice_5': [],
        'TRANGE': [],
        'rsi5': [], 'rsi10': [], 'rsi14': [],'mfi':[],
        'y': [],
    }  #v0版本
    stock_feature_v1 = {
        # 基本量价因子
        'open': [],
        'close': [],
        'high': [],
        'low': [],
        'next_open': [],
        'volume': [],
        'vwap': [],
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
        'next_open_percentage': [],
        'close_change': [],
        'open_change': [],
        'low_change': [],
        'high_change': [],
        'a_share_capital_percentage': [],
        'float_a_share_capital_percentage': [],
        'vwap_percentage': [],
        'vwap2close': [],
        'volume_change': [],
        'turnover_rate_change': [],
        'turnover_change': [],
        # # 重叠因子
        'EMA_5': [],
        'EMA_10': [],
        'EMA_20': [],
        'boll_upper': [],
        'boll_middle': [],
        'boll_lower': [],
        'mama': [],
        'fama': [],
        'sar': [],
        'dif': [],
        'dem': [],
        'histogram': [],
        'mom12': [],
        'mom26': [],
        # # #成交量因子
        # # # 'ADOSC':[],
        # # # 'obv':[],
        # # 波动性因子
        'natrPrice_5': [],
        'TRANGE': [],
        'rsi5': [], 'rsi10': [], 'rsi14': [], 'mfi': [],
        #特征yinzi
        'ema10_ema5':[],
        'ema20_ema5':[],
        'ema5_ema10':[],
        'ema5_ema20':[],
        'ema20_ema10':[],
        'ema10_ema20':[],
        # 'volume_rank':[],
        # 'turnover_rank':[],
        # 'turnover_rate_rank':[],
        'boll_upper_breakout':[],
        'boll_lower_breakout':[],
        # 'slowk':[],
        # 'slowd':[],
        # 'slowj':[],
        # 'turnover_rate_mid':[],
        # 'turnover_rate_std':[],
        # 'turnover_rate_up':[],
        #'volume_signal':[],
        #'type':[],
        'y': [],
    }
    input_dim = len(stock_feature_v1.keys())-1 # 假设输入包含开盘价、最高价、最低价、收盘价和成交量
    print('factor num:{}'.format(input_dim))
    output_dim = 1  # 输出是收益率
    d_model = 256
    nhead = 16
    num_encoder_layers = 4
    num_decoder_layers = 2
    dim_feedforward = 2048
    dropout = 0.1
    sequence_length = 10
    batch_size = 32
    num_epochs = 100
    window_size = 20
    global logger
    logger = logging_system('training_v1.log')

    # model = TransformerModel(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                          dim_feedforward, dropout).to(device)

    model = TransformerModel_reg(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                                     dim_feedforward, dropout).to(device)

    train_NAT(model,'./model_1211_{}factor/'.format(input_dim),stock_feature_v1,window_size,batch_size,num_epochs)
    #test(model,model_path='./model_2layer_16head_36factor/model_epoch22_0.0004_0.0188_0.0382_-0.1818.pth')
    #mse_loss,rmse_loss,corr,r2 = val(model,model_path='./model_2layer_16head_36factor/model_epoch22_0.0004_0.0188_0.0382_-0.1818.pth',training=False)
    # mse_loss, rmse_loss, corr, r2 = val_NAT(model, logger, 1, window_size=window_size, stock_feature=stock_feature,
    #                                         model_path='./model_2layer_16head_36factor/model_epoch22_0.0004_0.0188_0.0382_-0.1818.pth',
    #                                         training=False, plot_save_path='./model_2layer_16head_36factor/test', device=device)