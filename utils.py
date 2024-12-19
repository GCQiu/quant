import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from dataset import SlidingWindowDataset,SlidingWindowDataset_test
from models.transformer import TransformerModel,TransformerModel_NAT
import numpy as np
import logging
from scipy.stats import zscore
from sklearn.metrics import r2_score
from losses import weighted_mse_loss
import matplotlib.pyplot as plt
import mplfinance as mpf


def test_NAT(model,model_path,test_path,plot_save_path,stock_feature,window_size,if_save_plot=True,device=None,logger=None):
    #model.load_state_dict(torch.load('transformer_model.pth'))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    criterion_1 = nn.MSELoss()
    test_dataset = SlidingWindowDataset(test_path, window_size, None,stock_feature,is_training=False)
    whole_result = []
    if if_save_plot:
        plot_result = os.path.join(plot_save_path, 'test_result')
        os.makedirs(plot_result, exist_ok=True)
    test_result = {
        'mse_result':[],
        'rmse_result':[],
        'corre_result':[],
        'r2_result':[],
        'mse_result_zscore':[],
        'rmse_result_zscore':[],
        'corre_result_zscore':[],
        'r2_result_zscore':[]
    }
    gt_y_all = []
    predict_all = []
    all_profit = []
    returns = []
    all_sharpe = []
    test_list = open('good_stocks.txt','r').readlines()
    drawdown = []
    all_annualized_return = []
    all_returns = []
    for csv in tqdm.tqdm(test_list):
            df = pd.read_csv(os.path.join(test_path, csv.strip()), index_col=0)
            if df.shape[0] < window_size:
                continue
            else:
                y_non_zero = df[df['y'] == 0]
                y_num = y_non_zero.shape[0]
                if df.shape[0]-y_num < window_size:
                    window_size = df.shape[0]-y_num
                else:
                    window_size = window_size
                test_df = df.tail(y_num + window_size - 1).reset_index(drop=True)
                dec_input = df.tail(y_num + window_size - 1).copy().reset_index(drop=True)
            final_out = []

            for i in range(0,len(test_df) - window_size + 1,1):
                stocks_feature, y = test_dataset.test_preprocess_v1(test_df[i:i+window_size])
                src = torch.tensor(np.array(stocks_feature),dtype=torch.float32).unsqueeze(0).to(device) #N d L

                #tgt = torch.tensor(dec_input[i:i+window_size].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # N d L

                output = model.forward(src, src)
                final_out.append(output.item())
                #final_out_zscore.append(output_zscore.item())
                dec_input.loc[i+window_size-1, 'y'] = output.item()
            #df.loc[df['y']==0,'y'] = [i/100 for i in final_out]
            #print(df.tail(100))
            #df.to_csv(os.path.join('./final_result', csv.strip()))
            dec_input['3_y_EMA'] = 0
            dec_input['y_diff'] = 0
            dec_input.loc[window_size-1:,'3_y_EMA'] = dec_input.loc[window_size-1:,'y'].ewm(span=5, adjust=False).mean()
            #dec_input.loc[window_size-1:,'3_y_EMA'] = dec_input.loc[window_size-1:,'y'].rolling(window=5).mean()
            dec_input.loc[window_size - 1:, 'y_diff'] = dec_input.loc[window_size - 1:, '3_y_EMA'].diff()
            y_diff = list(dec_input['y_diff'][window_size-1:])
            # 找出上升趋势的一段
            ascending_segments = []
            start_index = None

            for i in range(5, len(y_diff)):
                if y_diff[i] > 0:
                    if start_index is None:
                        start_index = i
                else:
                    if start_index is not None:
                        ascending_segments.append((start_index, i - 1))
                        start_index = None

            # 处理最后一个上升趋势段
            if start_index is not None:
                ascending_segments.append((start_index, len(y_diff) - 1))

            print(ascending_segments)
            ema5 = list(dec_input['EMA_5'][window_size - 1:])
            ema10 = list(dec_input['EMA_10'][window_size - 1:])
            rsi14 = list(dec_input['rsi14'][window_size - 1:])
            dec_input['signal']=0
            cash = 0
            profit = [0 for i in range(y_num)]
            cash_change = []
            buy = []
            for i, pos in enumerate(ascending_segments):
                try:
                    if pos[1] - pos[0] > 1:
                        if ema5[pos[0]+1]-ema5[pos[0]] > 0:
                            buy_price = dec_input['open'][pos[0]+1]
                            cash += (dec_input['open'][pos[1]+1]-dec_input['open'][pos[0]+1])
                            dec_input.loc[pos[0]+1+window_size-1:pos[1]+1+window_size-1, 'signal'] = 1
                            #dec_input.loc[pos[1]+1+window_size-1, 'signal'] = -1
                            buy.append(pos)
                except:
                    print('no data')
            all_profit.append(cash)
            print(cash,buy)

            dec_input['Strategy_Returns'] = dec_input['open_change'] * dec_input['signal'].shift(1)
            dec_input['Cumulative_Returns'] = (1 + dec_input['open_change']).cumprod()
            dec_input['Cumulative_Strategy_Returns'] = (1 + dec_input['Strategy_Returns']).cumprod()
            print(dec_input.iloc[-1]['Cumulative_Strategy_Returns'])
            returns.append(dec_input.iloc[-1]['Cumulative_Strategy_Returns'])


            all_returns.append(list(dec_input['Cumulative_Returns'][window_size - 1:]))
            #print(all_returns)


            fig, ax = plt.subplots(4,1,figsize=(20, 15))
            ax[0].plot([i for i in range(y_num)], np.array(final_out), label='predict', color='red')
            ax[0].plot([i for i in range(y_num)], dec_input.loc[window_size-1:,'3_y_EMA'], label='y_ema7', color='blue')
            #ax[1].plot([i for i in range(y_num)], test_df['y'][window_size-1:]*100, label='y_zscore', color='blue')
            ax[1].plot([i for i in range(y_num)], list(dec_input['open'][window_size-1:]), label='open_price', color='black')
            ax[3].plot([i for i in range(y_num)], dec_input['Cumulative_Strategy_Returns'][window_size-1:], label='Cumulative_Strategy_Returns',
                       color='red')
            ax[3].plot([i for i in range(y_num)], dec_input['Cumulative_Returns'][window_size - 1:],
                       label='Cumulative_Returns',
                       color='blue')

            ema5 = list(dec_input['EMA_5'][window_size - 1:])
            ema10 = list(dec_input['EMA_10'][window_size - 1:])
            for i, pos in enumerate(buy):
               try:
                    ax[1].plot(pos[0]+1, list(dec_input['open'][window_size-1:])[pos[0]+1], 'ro', markersize=10)
                    ax[1].plot(pos[1]+1, list(dec_input['open'][window_size - 1:])[pos[1]+1], 'go', markersize=10)
               except:
                   print('no data')
            #ax[1].plot([i for i in range(y_num)], [0 for i in range(y_num)], label='predict', color='black')
            #ax[1].plot([i for i in range(100)], np.array(final_out) * std + mean, label='predict', color='red')
            df.rename(columns={'date': 'Date'}, inplace=True)
            # df.set_index('Date', inplace=True)
            df.index = pd.DatetimeIndex(df['Date'])
            #df.index = pd.to_datetime(df.index)

            #print(len(test_df['y'][window_size-1:]))
            mpf.plot(df.tail(y_num), type='candle', style='charles',ylabel='Price', ax=ax[2])
            ax[0].set_title(csv)
            ax[0].legend()
            ax[1].set_title(csv)
            ax[1].legend()
            ax[3].legend()
            plt.savefig(os.path.join(plot_result,csv.strip().replace('.csv','.jpg')),dpi=200)
            #
            # predict = torch.tensor(np.array(final_out),dtype=torch.float32).unsqueeze(0)
            # gt_y = torch.tensor(np.array(test_df['y'][window_size-1:]),dtype=torch.float32).unsqueeze(0)
            #
            # predict_zscore = torch.tensor(np.array(final_out_zscore),dtype=torch.float32).unsqueeze(0).to(device)
            # gt_y_zscore = torch.tensor(np.array(test_df['y_zscore'][19:].values),dtype=torch.float32).unsqueeze(0).to(device)
            #
            # mse_loss_zscore = criterion_1(predict_zscore, gt_y_zscore)
            # rmse_loss_zscore = torch.sqrt(mse_loss_zscore)
            # correlation_matrix_zscore = torch.corrcoef(torch.cat((predict_zscore, gt_y_zscore), dim=0))
            # test_result['corre_result_zscore'].append(correlation_matrix_zscore[1, 0].item())
            # test_result['mse_result_zscore'].append(mse_loss_zscore.item())
            # test_result['rmse_result_zscore'].append(rmse_loss_zscore.item())
            #
            # mse_loss = criterion_1(predict, gt_y)
            # rmse_loss = torch.sqrt(mse_loss)
            # correlation_matrix = torch.corrcoef(torch.cat((predict, gt_y), dim=0))
            #
            # gt_y_all.append(list(gt_y.numpy()))
            # predict_all.append(list(predict.numpy()))
            #
            # test_result['corre_result'].append(correlation_matrix[1,0].item())
            # test_result['mse_result'].append(mse_loss.item())
            # test_result['rmse_result'].append(rmse_loss.item())
            window_size = 20
    positive_count = len([num for num in all_profit if num > 0])

    negative_count = len([num for num in all_profit if num < 0])



    print('returns',sum(returns)/len(returns))
    returns = pd.Series(returns)
    annualized_return = (returns.mean() ** (252 / y_num) - 1)
    strategy_std = returns.std()
    sharpe_ratio = (annualized_return - 0.02) / strategy_std

    print('sharpe_ratio',sharpe_ratio)

    print('annu',annualized_return)
    print(positive_count,negative_count)

    quxian = []
    for i in range(y_num):
        temp = 0
        day_num = 0
        for j in range(len(all_returns)):
            try:
                temp+=all_returns[j][i]
            except:
                temp+=0
        quxian.append(temp/len(all_returns))
    quxian = pd.Series(quxian)
    drawdown = (quxian - quxian.cummax()) / quxian.cummax()
    max_drawdown = drawdown.min()
    print('drawdown', max_drawdown)
    plt.figure(figsize=(10, 5))
    plt.plot(quxian)
    plt.savefig('1.png',dpi=200)
    # r2 = r2_score(np.array(gt_y_all), np.array(predict_all))
    # logger.info('val mse loss : {:.4f}'.format(sum(test_result['mse_result'])/len(test_result['mse_result'])))
    # logger.info('val rmse loss : {:.4f}'.format(sum(test_result['rmse_result']) / len(test_result['rmse_result'])))
    # logger.info('correlation : {:.4f}'.format(sum(test_result['corre_result']) / len(test_result['corre_result'])))
    # logger.info('r2 : {:.4f}'.format(r2))
    # logger.info('val mse zscore loss : {:.4f}'.format(sum(test_result['mse_result_zscore']) / len(test_result['mse_result_zscore'])))
    # logger.info('val rmse zscore loss : {:.4f}'.format(sum(test_result['rmse_result_zscore']) / len(test_result['rmse_result_zscore'])))
    # logger.info('correlation zscore: {:.4f}'.format(sum(test_result['corre_result_zscore']) / len(test_result['corre_result_zscore'])))

    return sum(test_result['mse_result']) / len(test_result['mse_result']),sum(test_result['rmse_result']) / len(test_result['rmse_result']),\
            sum(test_result['corre_result']) / len(test_result['corre_result']),r2

def test_NAT_testset(model,model_path,test_path,plot_save_path,stock_feature,window_size,if_save_plot=True,device=None,logger=None):
    #model.load_state_dict(torch.load('transformer_model.pth'))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    criterion_1 = nn.MSELoss()
    test_dataset = SlidingWindowDataset(test_path, window_size, None,stock_feature,is_training=False)
    whole_result = []
    if if_save_plot:
        plot_result = os.path.join(plot_save_path, 'test_result')
        os.makedirs(plot_result, exist_ok=True)
    test_result = {
        'mse_result':[],
        'rmse_result':[],
        'corre_result':[],
        'r2_result':[],
        'mse_result_zscore':[],
        'rmse_result_zscore':[],
        'corre_result_zscore':[],
        'r2_result_zscore':[]
    }
    gt_y_all = []
    predict_all = []
    good_stocks = open('good_stocks.txt','a')
    #test_list = open('test_file.txt','r').readlines()
    for csv in tqdm.tqdm(os.listdir(test_path)):
        if csv in os.listdir('./dataset_zero_need_predict'):
            df = pd.read_csv(os.path.join(test_path, csv.strip()), index_col=0)
            test_df = df.tail(100 + window_size - 1).reset_index(drop=True)
            dec_input = df.tail(100 + window_size - 1).copy().reset_index(drop=True)
            dec_input.loc[window_size - 1:,'y'] = 0
            final_out = []
            for i in range(0,len(test_df) - window_size + 1,1):
                stocks_feature, y = test_dataset.test_preprocess_v1(test_df[i:i+window_size])
                src = torch.tensor(np.array(stocks_feature),dtype=torch.float32).unsqueeze(0).to(device) #N d L

                #tgt = torch.tensor(dec_input[i:i+window_size].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # N d L

                output = model.forward(src, src)
                final_out.append(output.item())
                #final_out_zscore.append(output_zscore.item())
                dec_input.loc[i+window_size-1, 'y'] = output.item()

            dec_input['3_y_EMA'] = 0
            dec_input['y_diff'] = 0
            dec_input.loc[window_size - 1:, '3_y_EMA'] = dec_input.loc[window_size - 1:, 'y'].ewm(span=5,
                                                                                                  adjust=False).mean()
            # dec_input.loc[window_size-1:,'3_y_EMA'] = dec_input.loc[window_size-1:,'y'].rolling(window=5).mean()
            dec_input.loc[window_size - 1:, 'y_diff'] = dec_input.loc[window_size - 1:, '3_y_EMA'].diff()
            y_diff = list(dec_input['y_diff'][window_size - 1:])
            # 找出上升趋势的一段
            ascending_segments = []
            start_index = None

            for i in range(5, len(y_diff)):
                if y_diff[i] > 0:
                    if start_index is None:
                        start_index = i
                else:
                    if start_index is not None:
                        ascending_segments.append((start_index, i - 1))
                        start_index = None

            # 处理最后一个上升趋势段
            if start_index is not None:
                ascending_segments.append((start_index, len(y_diff) - 1))

            print(ascending_segments)
            ema5 = list(dec_input['EMA_5'][window_size - 1:])
            ema10 = list(dec_input['EMA_10'][window_size - 1:])
            rsi14 = list(dec_input['rsi14'][window_size - 1:])
            cash = 0
            cash_change = []
            buy = []
            for i, pos in enumerate(ascending_segments):
                try:
                    if pos[1] - pos[0] > 1:
                        if ema5[pos[0] + 1] - ema5[pos[0]] > 0 and rsi14[pos[0] + 1] < 70:
                            buy_price = dec_input['open'][pos[0] + 1]
                            cash += (dec_input['open'][pos[1] + 1] - dec_input['open'][pos[0] + 1])
                            buy.append(pos)
                except:
                    print('no data')
            if cash>=0:
                good_stocks.write(csv+'\n')

            # fig, ax = plt.subplots(4,1,figsize=(20, 15))
            # ax[0].plot([i for i in range(100)], np.array(final_out), label='predict', color='red')
            # ax[1].plot([i for i in range(100)], test_df['y'][window_size-1:], label='y', color='black')
            # ax[2].plot([i for i in range(100)], test_df['y'][window_size - 1:]*100, label='y', color='black')
            # ax[2].plot([i for i in range(100)], np.array(final_out), label='predict', color='red')
            # #ax[1].plot([i for i in range(y_num)], [0 for i in range(y_num)], label='predict', color='black')
            # #ax[1].plot([i for i in range(100)], np.array(final_out) * std + mean, label='predict', color='red')
            # df.rename(columns={'date': 'Date'}, inplace=True)
            # # df.set_index('Date', inplace=True)
            # df.index = pd.DatetimeIndex(df['Date'])
            # #df.index = pd.to_datetime(df.index)
            #
            # #print(len(test_df['y'][window_size-1:]))
            # mpf.plot(df.tail(100), type='candle', style='charles',ylabel='Price', ax=ax[3])
            # ax[0].set_title(csv)
            # ax[0].legend()
            # ax[1].legend()
            # ax[2].legend()
            # plt.savefig(os.path.join(plot_result,csv.strip().replace('.csv','.jpg')),dpi=200)
            #
            # predict = torch.tensor(np.array(final_out)/100,dtype=torch.float32).unsqueeze(0)
            # gt_y = torch.tensor(np.array(test_df['y'][window_size-1:]),dtype=torch.float32).unsqueeze(0)

            # predict_zscore = torch.tensor(np.array(final_out_zscore),dtype=torch.float32).unsqueeze(0).to(device)
            # gt_y_zscore = torch.tensor(np.array(test_df['y_zscore'][19:].values),dtype=torch.float32).unsqueeze(0).to(device)
            #
            # mse_loss_zscore = criterion_1(predict_zscore, gt_y_zscore)
            # rmse_loss_zscore = torch.sqrt(mse_loss_zscore)
            # correlation_matrix_zscore = torch.corrcoef(torch.cat((predict_zscore, gt_y_zscore), dim=0))
            # test_result['corre_result_zscore'].append(correlation_matrix_zscore[1, 0].item())
            # test_result['mse_result_zscore'].append(mse_loss_zscore.item())
            # test_result['rmse_result_zscore'].append(rmse_loss_zscore.item())

            # mse_loss = criterion_1(predict, gt_y)
            # rmse_loss = torch.sqrt(mse_loss)
            # correlation_matrix = torch.corrcoef(torch.cat((predict, gt_y), dim=0))
            #
            # gt_y_all.append(list(gt_y.numpy()))
            # predict_all.append(list(predict.numpy()))
            #
            # test_result['corre_result'].append(correlation_matrix[1,0].item())
            # test_result['mse_result'].append(mse_loss.item())
            # test_result['rmse_result'].append(rmse_loss.item())
            window_size = 20
    r2 = r2_score(np.squeeze(np.array(gt_y_all)), np.squeeze(np.array(predict_all)))
    logger.info('val mse loss : {:.4f}'.format(sum(test_result['mse_result'])/len(test_result['mse_result'])))
    logger.info('val rmse loss : {:.4f}'.format(sum(test_result['rmse_result']) / len(test_result['rmse_result'])))
    logger.info('correlation : {:.4f}'.format(sum(test_result['corre_result']) / len(test_result['corre_result'])))
    logger.info('r2 : {:.4f}'.format(r2))

    # logger.info('val mse zscore loss : {:.4f}'.format(sum(test_result['mse_result_zscore']) / len(test_result['mse_result_zscore'])))
    # logger.info('val rmse zscore loss : {:.4f}'.format(sum(test_result['rmse_result_zscore']) / len(test_result['rmse_result_zscore'])))
    # logger.info('correlation zscore: {:.4f}'.format(sum(test_result['corre_result_zscore']) / len(test_result['corre_result_zscore'])))

    return sum(test_result['mse_result']) / len(test_result['mse_result']),sum(test_result['rmse_result']) / len(test_result['rmse_result']),\
            sum(test_result['corre_result']) / len(test_result['corre_result']),r2

def val_NAT(model,logger,epoch,model_path,training,plot_save_path,stock_feature,window_size,if_save_plot=True,device=None):
    #model.load_state_dict(torch.load('transformer_model.pth'))
    if training:
        model.eval()
    else:
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.L1Loss()
    test_path = './dataset_test_v0'
    test_dataset = SlidingWindowDataset(test_path, window_size, None,stock_feature,is_training=False)
    whole_result = []
    if if_save_plot:
        plot_result = os.path.join(plot_save_path, 'plot_{}'.format(epoch))
        os.makedirs(plot_result, exist_ok=True)
    test_result = {
        'mse_result':[],
        'rmse_result':[],
        'corre_result':[],
        'r2_result':[],
        'mse_result_zscore':[],
        'rmse_result_zscore':[],
        'corre_result_zscore':[],
        'r2_result_zscore':[]
    }
    gt_y_all = []
    predict_all = []
    test_list = open('test_file.txt','r').readlines()
    for csv in tqdm.tqdm(test_list):
            df = pd.read_csv(os.path.join(test_path, csv.strip()), index_col=0)

            test_df = df.tail(100+window_size-1).reset_index(drop=True)
            dec_input = df['y'].tail(100+window_size-1).copy().reset_index(drop=True)
            dec_input.loc[window_size-1:] = 0
            #dec_input[0:19] = zscore(dec_input[0:19])
            final_out = []
            mean = dec_input[0:window_size].mean()
            std = dec_input[0:window_size].std()
            dec_input[0:window_size] = (dec_input[0:window_size]-mean)/std
            # max_val = max(dec_input)
            # min_val = min(dec_input)
            # dec_input[0:19] = (dec_input[0:19]-min_val)/(max_val-min_val)
            for i in range(0,len(test_df) - window_size + 1,1):
                stocks_feature, y = test_dataset.test_preprocess_v1(test_df[i:i+window_size])
                src = torch.tensor(np.array(stocks_feature),dtype=torch.float32).unsqueeze(0).to(device) #N d L

                tgt = torch.tensor(dec_input[i:i+window_size].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # N d L

                output = model.forward(src, src)
                final_out.append(output.item())
                #final_out_zscore.append(output_zscore.item())
                dec_input[i+window_size-1] = output.item()
            fig, ax = plt.subplots(3,1,figsize=(20, 15))
            ax[0].plot([i for i in range(100)], np.array(final_out), label='predict', color='red')
            ax[0].plot([i for i in range(100)], test_df['y'][window_size-1:], label='y', color='blue')
            ax[1].plot([i for i in range(100)], test_df['y'][window_size-1:]*100, label='y_zscore', color='blue')
            ax[1].plot([i for i in range(100)], np.array(final_out), label='predict', color='green')
            ax[1].plot([i for i in range(100)], [0 for i in range(100)], label='predict', color='black')
            #ax[1].plot([i for i in range(100)], np.array(final_out) * std + mean, label='predict', color='red')
            df.rename(columns={'date': 'Date'}, inplace=True)
            # df.set_index('Date', inplace=True)
            df.index = pd.DatetimeIndex(df['Date'])
            #df.index = pd.to_datetime(df.index)

            #print(len(test_df['y'][window_size-1:]))
            mpf.plot(df.tail(100), type='candle', style='charles',ylabel='Price', ax=ax[2])
            ax[0].set_title(csv)
            ax[0].legend()
            ax[1].set_title(csv)
            ax[1].legend()
            plt.savefig(os.path.join(plot_result,csv.strip().replace('.csv','.jpg')),dpi=200)

            predict = torch.tensor(np.array(final_out)*std+mean,dtype=torch.float32).unsqueeze(0)
            gt_y = torch.tensor(np.array(test_df['y'][window_size-1:]),dtype=torch.float32).unsqueeze(0)

            # predict_zscore = torch.tensor(np.array(final_out_zscore),dtype=torch.float32).unsqueeze(0).to(device)
            # gt_y_zscore = torch.tensor(np.array(test_df['y_zscore'][19:].values),dtype=torch.float32).unsqueeze(0).to(device)
            #
            # mse_loss_zscore = criterion_1(predict_zscore, gt_y_zscore)
            # rmse_loss_zscore = torch.sqrt(mse_loss_zscore)
            # correlation_matrix_zscore = torch.corrcoef(torch.cat((predict_zscore, gt_y_zscore), dim=0))
            # test_result['corre_result_zscore'].append(correlation_matrix_zscore[1, 0].item())
            # test_result['mse_result_zscore'].append(mse_loss_zscore.item())
            # test_result['rmse_result_zscore'].append(rmse_loss_zscore.item())

            mse_loss = criterion_1(predict, gt_y)
            rmse_loss = torch.sqrt(mse_loss)
            correlation_matrix = torch.corrcoef(torch.cat((predict, gt_y), dim=0))

            gt_y_all.append(list(gt_y.numpy()))
            predict_all.append(list(predict.numpy()))

            test_result['corre_result'].append(correlation_matrix[1,0].item())
            test_result['mse_result'].append(mse_loss.item())
            test_result['rmse_result'].append(rmse_loss.item())
    r2 = r2_score(np.squeeze(np.array(gt_y_all)), np.squeeze(np.array(predict_all)))
    logger.info('val mse loss : {:.4f}'.format(sum(test_result['mse_result'])/len(test_result['mse_result'])))
    logger.info('val rmse loss : {:.4f}'.format(sum(test_result['rmse_result']) / len(test_result['rmse_result'])))
    logger.info('correlation : {:.4f}'.format(sum(test_result['corre_result']) / len(test_result['corre_result'])))
    logger.info('r2 : {:.4f}'.format(r2))
    # logger.info('val mse zscore loss : {:.4f}'.format(sum(test_result['mse_result_zscore']) / len(test_result['mse_result_zscore'])))
    # logger.info('val rmse zscore loss : {:.4f}'.format(sum(test_result['rmse_result_zscore']) / len(test_result['rmse_result_zscore'])))
    # logger.info('correlation zscore: {:.4f}'.format(sum(test_result['corre_result_zscore']) / len(test_result['corre_result_zscore'])))

    return sum(test_result['mse_result']) / len(test_result['mse_result']),sum(test_result['rmse_result']) / len(test_result['rmse_result']),\
            sum(test_result['corre_result']) / len(test_result['corre_result']),r2

def scale_to_range(column):
    min_val = np.min(column)
    max_val = np.max(column)
    return (column - min_val) / (max_val - min_val)

def inverse_scale_to_range(column,min_val,max_val):
    column = ((column+1)/2)*(max_val - min_val)+min_val
    return column