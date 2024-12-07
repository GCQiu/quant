import torch
import argparse
import os
import logging
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
print('local_rank',local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


phone_set_40 = ['SIL','AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
phone2vism_dict = {'sp': 'SIL', 'sp_sil': 'SIL', 'sil': 'SIL', 'SIL': 'SIL', 'spn': 'SIL', 'AA': 'AA', 'AH': 'AA', 'AO': 'AA', 'AW': 'AA', 'HH': 'AA', 'AE': 'AE', \
                   'EY': 'AE', 'EH': 'AE', 'ER': 'AE', 'AY': 'AE', 'M': 'MBP', 'B': 'MBP', 'P': 'MBP', 'F': 'FV', 'V': 'FV', 'TH': 'TH', 'D': 'TH', 'DH': 'TH', 'T': \
                       'TH', 'L': 'TH', 'G': 'TH', 'K': 'TH', 'S': 'S', 'Z': 'S', 'IY': 'S', 'IH': 'S', 'Y': 'S', 'UW': 'UW', 'UH': 'UW', 'W': 'UW', 'SH': 'SH', 'CH': \
                       'SH', 'JH': 'SH', 'ZH': 'SH', 'R': 'SH', 'OW': 'OW', 'OY': 'OW', 'N': 'NG', 'NG': 'NG'}
frame_list = { "SIL":0,"AA": 1, "OW": 2, "AE": 3, "S": 4, "UW": 5, "FV": 6, "MBP": 7, "SH": 8, "NG": 9, "TH": 10}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.queue = np.zeros(shape=[100, ])
        self.index = 0
        self.queue_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.queue[self.index % 100] = self.val
        self.index += 1
        self.queue_avg = np.mean(self.queue)
def check_parameters(model):
    # check the parameters
    for param in model.parameters():
        print(param.shape, param.requires_grad)
def load_checkpoint(model):
    start_epoch = 0
    best_loss = 1e6
    if args.pretrained_model:

        if not os.path.isfile(args.pretrained_model):
            logger.info("=> no checkpoint found at '{}'".format(args.pretrained_model))
            return
        logger.info('=> load checkpoint {}'.format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        model_dict = model.state_dict()

        #print(predict.keys())
        #print(checkpoint.keys())
        pretrained_dict = {k.replace('module.',''):v for k, v in checkpoint['state_dict'].items() }
        #check_parameters(model)
        model_dict.update(pretrained_dict)
        #print(pretrained_dict.keys())
        #print(model_dict)
        if 'epoch' in checkpoint.keys():
            start_epoch = checkpoint['epoch']
        if 'loss' in checkpoint.keys():
            best_loss = checkpoint['loss']
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(model_dict)
        # if 'network' in checkpoint.keys():
        #     model.load_state_dict(checkpoint['network'])
        logger.info(
            '=> loaded checkpoint {} (epoch {}) (best_loss {})'.format(args.pretrained_model, start_epoch, best_loss))
    return start_epoch, best_loss



def save_checkpoint(model, epoch, loss, best_loss):
    os.makedirs(args.save_path, exist_ok=True)
    model_path = os.path.join(args.save_path, 'ckpt_epoch%d.pth' % epoch)
    # torch.save({
    #     'epoch': epoch + 1,
    #     'loss': loss,
    #     'state_dict': model.state_dict(), }, model_path)
    model_path = os.path.join(args.save_path, 'best_ckpt.pth')
    if loss < best_loss or not os.path.exists(model_path):
        if torch.distributed.get_rank()==0:
            best_loss = loss
            torch.save({
                'epoch': epoch + 1,
                'loss': best_loss,
                'state_dict': model.state_dict()}, model_path)
    return best_loss


def logging_system():
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s] %(message)s')

    sysh = logging.StreamHandler()
    sysh.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(args.save_path, args.logger), 'w')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sysh)
    return logger


def train(train_loader, model, criterion, optimizer, epoch, log_writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    logger.info("Training....")
    model.train()
    end = time.time()
    for iter_batch,(waveform, y_visms) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        feature_batch = waveform.cuda().long()
        y_vism_batch = y_visms.cuda().float()

        output = model(feature_batch,hidden=None)
        # if output.size(1)<y_vism_batch.size(1):
        #     diff = y_vism_batch.size(1)-output.size(1)
        #     output = F.pad(output,(0,0,0,diff))
        # else:
        #     diff =  output.size(1) - y_vism_batch.size(1)
        #     y_vism_batch = F.pad(y_vism_batch, (0, 0, 0, diff))
        #print(output.shape,y_vism_batch.shape,'xxxxxxxxxxx')

        loss = criterion(output , y_vism_batch)
        log_writer.add_scalar('Train_loss', loss, epoch * len(train_loader) + iter_batch)
        #print('Train_loss', loss)
        losses.update(loss.item(), feature_batch.size(0))
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_batch % args.log_freq == 0:
            lr = args.learning_rate * (0.1 ** (int(epoch / args.lr_decay_step)))
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} [{loss.queue_avg:.4f},{loss.avg:.4f}]\t'
                        '{lr:.4f}'.format(
                epoch, iter_batch, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,lr=lr))


def val(val_loader, model, criterion, log_writer, epoch):
    losses = AverageMeter()
    logger.info("valing....")
    model.eval()

    for iter_batch,(waveform, y_visms) in enumerate(val_loader):
        feature_batch = waveform.cuda().long()
        y_vism_batch = y_visms.cuda().float()
        # print(img_batch.shape, label_batch.shape)
        output = model(feature_batch)
        # if output.size(1) < y_vism_batch.size(1):
        #     diff = y_vism_batch.size(1) - output.size(1)
        #     output = F.pad(output, (0, 0, 0, diff))
        # else:
        #     diff = output.size(1) - y_vism_batch.size(1)
        #     y_vism_batch = F.pad(y_vism_batch, (0, 0, 0, diff))
        loss = criterion(output , y_vism_batch)
        losses.update(loss.item(), feature_batch.size(0))

    logger.info('Loss {loss.avg:.8f}\t'.format(loss=losses,))

    if log_writer != None:
        log_writer.add_scalar('val_loss', losses.avg, epoch)

    return losses.avg


def adjust_learning_rate(optimizer, epoch, log_writer):
    lr = args.learning_rate * (0.1 ** (int(epoch / args.lr_decay_step)))
    # lr = args.learning_rate * (0.1**(epoch/10))
    lr = args.min_lr if lr <= args.min_lr else lr
    if epoch % (args.lr_decay_step) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    logger.info('learning rate: %f' % lr)
    log_writer.add_scalar('learning_rate', lr, epoch)


def main():
    from dataset.phone_dataset import BatchPhoneDataset as dataset
    from models.phone2vism import TCN_GRU_res_jap,TCN_GRU_res_kor


    train_dst = dataset(audio_list_files=args.train_list)
    train_sampler = DistributedSampler(train_dst)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                              pin_memory=True,sampler=train_sampler,collate_fn=train_dst.collate_fill)

    val_dst = dataset(audio_list_files=args.train_list)
    test_sampler = DistributedSampler(val_dst)
    val_loader = DataLoader(val_dst, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False,sampler=test_sampler,collate_fn=train_dst.collate_fill)

    start_epoch = 0
    best_loss = 1e6

    model = TCN_GRU_res_jap()
    # for name, param in model.named_parameters():
    #     if 'wav_model' in name:
    #         param.requires_grad=False

    start_epoch, best_loss = load_checkpoint(model)
    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss(reduction='mean')
    criterion = nn.L1Loss()
    # if torch.cuda.is_available():
    #     model = nn.DataParallel(model).cuda()
    #     criterion = criterion.cuda()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,find_unused_parameters=True)


    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    log_writer = SummaryWriter(os.path.join(args.save_path, 'Training_log'))
    # loss = val(val_loader, model, criterion, None, None)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch, log_writer)

        train(train_loader, model, criterion, optimizer, epoch, log_writer)
        loss = val(val_loader, model, criterion, log_writer, epoch)
        best_loss = save_checkpoint(model, epoch, loss, best_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train scene phone2vism')
    parser.add_argument('--window_size', default=300, type=int, help='input sequence len')
    parser.add_argument('--gpus', default='0', type=str, help='identify gpus')
    parser.add_argument('--workers', '-j', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs to run')
    parser.add_argument('--batch_size', '-b', default=128, type=int,
                        help='mini batch')
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--logger', default='training.log', type=str,
                        help='log file')
    parser.add_argument('--log_freq', '-l', default=10, type=int,
                        help='print log')
    parser.add_argument('--train_list', default="/mnt/kaiwu-group-z3/cliffordqiu/Dataset/jvs_ver1_train.txt", type=str,
                        help='path to training list')
    parser.add_argument('--val_list', default="/mnt/kaiwu-group-z3/cliffordqiu/Dataset/jvs_ver1_test.txt", type=str,
                        help='path to validation list')
    parser.add_argument('--train_rootdir', default='', type=str,
                        help='the rootdir of train dataset')
    parser.add_argument('--val_rootdir', default='', type=str,
                        help='the rootdir of validation dataset')
    parser.add_argument('--save_path', default='./model_jap_phone2vism_align_revise/', type=str,
                        help='path to save checkpoint and log')
    parser.add_argument('--pretrained_model', default='', type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--phones_num', type=int, default=40, help='num phones')
    parser.add_argument('--vism_num', type=int, default=10, help='num classes')
    parser.add_argument('--lr_decay_step', type=int, default=10, help='reduce the learning rate every step')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='the min learning rate')
    parser.add_argument('--nproc_per_node', type=int, default=6, help='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--master_port', type=int, default=0)

    global args, logger, DCFNet


    args = parser.parse_args()
    logger = logging_system()
    logger.info(vars(args))
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # if torch.cuda.device_count() > 1:
    #     logger.info('%d GPU found' % torch.cuda.device_count())
    main()
