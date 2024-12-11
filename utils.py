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


def val(model,logger,model_path,training,plot_save_path ,if_save_plot=True,device=None):
    #model.load_state_dict(torch.load('transformer_model.pth'))
    if training:
        model.eval()
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.L1Loss()
    test_path = './dataset_test_v0'
    test_dataset = SlidingWindowDataset(test_path, 30, None,None,is_training=False)
    whole_result = []
    if if_save_plot:
        plot_result = os.path.join(plot_save_path, 'plot')
        os.makedirs(plot_result, exist_ok=True)
    for csv in tqdm.tqdm(os.listdir(test_path)[:50]):
            mse_result = []
            rmse_result = []
            corre_result = []
            # if '104060600017' in csv:
            df = pd.read_csv(os.path.join(test_path, csv), index_col=0)
            gt_y = df['y'].copy().tail(120).reset_index(drop=True)
            test_df = df.tail(120).reset_index(drop=True)
            dec_input = df['y'].tail(120).copy().reset_index(drop=True)
            dec_input.loc[20:] = 0
            for i in range(0,len(test_df) - 25 + 1,5):
                stocks_feature, y = test_dataset.test_preprocess_v1(test_df[i:i+25])
                src = torch.tensor(stocks_feature,dtype=torch.float32).unsqueeze(0).to(device)
                src = src[:, :, :20]
                final_out = []
                for j in range(5):
                    if j==0:
                        out = torch.tensor(dec_input[i:i+20].values, dtype=torch.float32).unsqueeze(1).unsqueeze(1).to(device)
                    output = model.predict(src, out, day_num=29)
                    final_out.append(output[-1, :, :].item())
                    out = torch.cat((out,output[-1,:,:].view(1,1,1)),dim=0)

                dec_input[i+20:i+20+5] = final_out
            fig, ax = plt.subplots()
            ax.plot([i for i in range(100)], dec_input[20:], label='predict', color='red')
            ax.plot([i for i in range(100)], gt_y[20:], label='gt', color='blue')
            ax.set_title(csv)
            plt.savefig(os.path.join(plot_result,csv.replace('.csv','.jpg')),dpi=200)

            predict = torch.tensor(np.array(dec_input[20:].values),dtype=torch.float32).unsqueeze(0).to(device)
            gt_y = torch.tensor(np.array(gt_y[20:].values),dtype=torch.float32).unsqueeze(0).to(device)
            mse_loss = criterion_1(predict, gt_y)
            rmse_loss = torch.sqrt(mse_loss)
            correlation_matrix = torch.corrcoef(torch.cat((predict, gt_y), dim=0))
            corre_result.append(correlation_matrix[1,0].item())
            mse_result.append(mse_loss.item())
            rmse_result.append(rmse_loss.item())
            logger.info('val mse loss : {:.4f}'.format(sum(mse_result)/len(mse_result)))
            logger.info('val rmse loss : {:.4f}'.format(sum(rmse_result) / len(rmse_result)))
            logger.info('correlation : {:.4f}'.format(sum(corre_result) / len(corre_result)))
            whole_result.append(rmse_loss)
    return sum(whole_result)/len(whole_result)

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