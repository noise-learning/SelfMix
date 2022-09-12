import datetime
import numpy as np
import random
import progressbar as pbar
import torch
import torch.nn.functional as F
from progressbar import DynamicMessage as DM


def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')


def count_noise(labeled_loader, unlabeled_loader):
   
    data1=labeled_loader.dataset
    data2=unlabeled_loader.dataset
    noise1=(data1.label != data1.cl)
    noise2=(data2.label != data2.cl)
    noise1 = noise1.nonzero()
    noise2 = noise2.nonzero()

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'labeled: %d  noise: %d ; unlabeled: %d  noise: %d ;ratio: %.4f'
        %(len(data1), len(noise1), len(data2), len(noise2), len(data2)/(len(data1)+len(data2))))


def get_progressbar(epoch,epochs, total, note):
    widgets = [pbar.CurrentTime(),' ',
        note,' ',str(epoch),'|',str(epochs),' ',
        pbar.Percentage(), ' ', pbar.Bar('#'), ' ', 
        DM('loss'),' ',DM('acc'),' ',DM('recall'),' ',
        pbar.Timer(), ' ', pbar.ETA(), pbar.FileTransferSpeed()]
    bar = pbar.ProgressBar(widgets=widgets, maxval=total)

    return bar


def get_co_progressbar(epoch, epochs, total, note):
    widgets = [pbar.CurrentTime(),' ',
        note,' ',str(epoch),'|',str(epochs),' ',
        pbar.Percentage(), ' ', pbar.Bar('#'), ' ', 
        DM('loss1'),' ',DM('acc1'),' ',
        DM('loss2'),' ',DM('acc2'),' ',
        pbar.Timer(), ' ', pbar.ETA(), pbar.FileTransferSpeed()]
    bar = pbar.ProgressBar(widgets=widgets, maxval=total)
    
    return bar