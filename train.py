import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from dataset import SlidingWindowDataset,SlidingWindowDataset_test
from models.transformer import TransformerModel,TransformerModel_NAT,TransformerModel_cls_reg,TransformerModel_gru,TransformerModel_reg
import numpy as np
import logging
from scipy.stats import zscore
from losses import weighted_mse_loss
import matplotlib.pyplot as plt
from utils import val,val_NAT
from tensorboardX import SummaryWriter

def logging_system(log_file):
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s] %(message)s')

    sysh = logging.StreamHandler()
    sysh.setFormatter(formatter)

    fh = logging.FileHandler(log_file, 'w')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sysh)
    return logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,model_path,stock_feature):
    batch_size = 64
    num_epochs = 50

    folder_path = './dataset_train_v0'
    os.makedirs(model_path, exist_ok=True)
    window_size = 25

    # 创建数据集和数据加载器
    dataset = SlidingWindowDataset(folder_path, window_size, batch_size,stock_feature,is_training=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    criterion_1 = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)



    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for i, (src, tgt) in enumerate(train_dataloader):
            #print(src.size(), tgt.size())
            optimizer.zero_grad()
            src = src.to(device)  #N d L
            tgt = tgt.to(device)  #N d L
            #print(src.size(), tgt.size())
            #start = torch.zeros((tgt.shape[0],tgt.shape[1],1)).to(device)
            output = model(src,src)  # L N d
            #output = output.permute(1, 2, 0)
            #loss = weighted_mse_loss(output, tgt[:,:,1:25],weight=0.6)
            loss = criterion_1(output, tgt[:,:,-1].unsqueeze(1))
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                logger.info(
                    f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                print(output[0,:,:],tgt[0,:,-1])
        val_loss = val(model,logger,model_path=None,training=True,plot_save_path = model_path,device=device)
        scheduler.step(val_loss)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), './{}/transformer_epoch{}_{:.4f}.pth'.format(model_path,str(epoch), val_loss))

    # 保存模型


def train_NAT(model,model_path,stock_feature):
    batch_size = 64
    num_epochs = 30

    folder_path = './dataset_train_v0'
    os.makedirs(model_path, exist_ok=True)
    window_size = 20


    log_path = os.path.join(model_path, 'log')
    os.makedirs(log_path, exist_ok=True)
    #writer = SummaryWriter(log_path)

    # 创建数据集和数据加载器
    dataset = SlidingWindowDataset(folder_path, window_size, batch_size,stock_feature,is_training=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    criterion_1 = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True,min_lr=0.00001,)




    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for i, (src, tgt) in enumerate(train_dataloader):
            #print(src.size(), tgt.size())
            optimizer.zero_grad()
            src = src.to(device)  #N d L
            tgt = tgt.to(device)  #N d L
            #print(src.size(), tgt.size())
            #start = torch.zeros((tgt.shape[0],tgt.shape[1],1)).to(device)
            output = model(src,tgt[:,:,:-1])  # L N d
            #output = output.permute(0, 2, 1)
            #loss = weighted_mse_loss(output, tgt[:,:,1:25],weight=0.6)
            loss = criterion_1(output, tgt[:,:,-1])
            #writer.add_scalar('training loss', loss, epoch * len(train_dataloader) + i)
            #loss = torch.sqrt(loss)
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            if (i + 1) % 20 == 0:
                logger.info(
                    f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.5f}')
                print(output[0,:].item(),tgt[0,:,-1].item())
                print(output[1, :].item(), tgt[1, :, -1].item())
        val_loss = val_NAT(model,logger,epoch,model_path=None,training=True,plot_save_path = model_path,device=device)
        scheduler.step(val_loss)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), './{}/transformer_epoch{}_{:.4f}.pth'.format(model_path,str(epoch), val_loss))


# 加载模型


# 预测函数
def test(model,model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.L1Loss()
    test_path = './dataset_test_v0'
    test_dataset = SlidingWindowDataset(test_path, 30, None,None,is_training=False)
    for csv in tqdm.tqdm(os.listdir(test_path)):
      if '104070000666.csv' in csv:
        csc_result = []
        corre_result = []
        # if '104060600017' in csv:
        df = pd.read_csv(os.path.join(test_path, csv), index_col=0)
        gt_y = df['y_scaled'].copy().tail(100).reset_index(drop=True)
        df = df.tail(129).reset_index(drop=True)
        #df.loc[29:, 'y_scaled'] = 0
        # print(df.loc[28,'y'],df.loc[29,'y'])
        for i in range(len(df) - 30 + 1):
          if i==28:
              for j in range(30):
                if j==0:
                    out = [0]
                stocks_feature, y = test_dataset.test_preprocess_v1(df[i:i + 30])
                # print(stocks_feature.shape, y.shape)
                src = torch.tensor(stocks_feature, dtype=torch.float32).unsqueeze(0).to(device)
                tgt = torch.tensor(out, dtype=torch.float32).unsqueeze(0).to(device)
                start = torch.zeros((tgt.shape[0], 1)).to(device) #N L

                output = model.predict(src, torch.cat((start,tgt[:,:-1]),dim=-1), day_num=29)
                print(output[0, :, :].item(),y[-1],gt_y[j])
                out.append(output[-1, :, :].item())
                #df.loc[i + 29, 'y_scaled'] = output[-1, :, :].item()

        #print(len(df['y'].tail(100)),len(gt_y.tail(100)))
        print(df['y_scaled'].tail(100),gt_y.tail(100))
        predict = torch.tensor(df['y_scaled'].tail(100).values, dtype=torch.float32)
        gt = torch.tensor(gt_y.values, dtype=torch.float32)
        loss = criterion_1(predict, gt)
        print(loss.item())
        correlation_matrix = torch.corrcoef(torch.stack((predict, gt)))
        corre_result.append(correlation_matrix[1, 0].item())
        # print(correlation_matrix)
        csc_result.append(loss.item())
    logger.info('val loss : {:.4f}'.format(sum(csc_result) / len(csc_result)))
    logger.info('correlation : {:.4f}'.format(sum(corre_result) / len(corre_result)))
    return sum(csc_result) / len(csc_result)
    #return sum(csc_result) / len(csc_result)






if __name__ == '__main__':
    input_dim = 19 # 假设输入包含开盘价、最高价、最低价、收盘价和成交量
    output_dim = 1  # 输出是收益率
    d_model = 256
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 512
    dropout = 0.2
    sequence_length = 10
    batch_size = 32
    num_epochs = 50
    stock_feature = {
            'open':[],
            'close':[],
            'next_open':[],
            'EMA_5':[],
            'EMA_10':[],
            'EMA_20':[],
            'a_share_capital':[],
            'a_share_capital_percentage': [],
            'float_a_share_capital':[],
            'float_a_share_capital_percentage': [],
            'rsi5':[],
            'rsi10': [],
            'rsi14': [],
            'Return': [],
            'EMA_5_trend': [],
            'EMA_10_trend': [],
            'EMA_20_trend': [],
            'pseudo_y': [],
            'volume': [],
            'turnover_rate': [],
            'turnover': [],
            'type': [],
            'y':[],
            #'y_scaled': [],
            'y_zscore': [],
            #'y_normalized':[]
        }
    global logger
    logger = logging_system('training.log')

    # model = TransformerModel(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
    #                          dim_feedforward, dropout).to(device)

    model = TransformerModel_reg(input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers,
                                                     dim_feedforward, dropout).to(device)

    train_NAT(model,'./model_3layer_8head_19feature_v1/',stock_feature)
    #test(model,model_path='./model_6layer_13feature/transformer_epoch1_0.0492.pth')
    #loss = val(model,model_path='./model_6layer_13feature/transformer_epoch0_0.0004.pth',training=False)
