import datetime

import numpy as np
import progressbar as pbar
import torch.nn.functional as F
from progressbar import DynamicMessage as DM
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')


def get_progressbar(epoch,epochs, total, note):
    widgets = [pbar.CurrentTime(),' ',
        note,' ',str(epoch),'|',str(epochs),' ',
        pbar.Percentage(), ' ', pbar.Bar('#'), ' ', 
        DM('loss'),' ',DM('acc'),' ',DM('recall'),' ',
        pbar.Timer(), ' ', pbar.ETA(), pbar.FileTransferSpeed()]
    bar = pbar.ProgressBar(widgets=widgets, maxval=total)

    return bar


def count_noise(labeled_loader, unlabeled_loader):
   
    data1=labeled_loader.dataset
    data2=unlabeled_loader.dataset
    noise1=(data1.label != data1.cl)
    noise2=(data2.label != data2.cl)
    noise1 = noise1.nonzero()
    noise2 = noise2.nonzero()

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'labeled: %d  noise: %d ; unlabeled: %d  noise: %d ;ratio: %.4f'
        %(len(data1), len(noise1), len(data2), len(noise2), len(data2)/(len(data1)+len(data2))))


def count_noise2(args, train_loader, cl_idx, cl_idx2):
    data=train_loader.dataset

    noise_lst=[]
    noise_num=0
    for id in range(args.num_class):

        noise=(data.label != (data.cl * cl_idx2[id]))*cl_idx[id]
        noise=noise.nonzero()
        noise_num += len(noise)
        noise_lst.append(len(noise)/len(cl_idx[id].nonzero()))

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), noise_lst, noise_num/len(data))


def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def metric(preds,labels):
    pred_flat=np.argmax(preds,axis=1).flatten()
    labels_flat=labels.flatten()
    return accuracy_score(labels_flat,pred_flat),\
        precision_score(labels_flat,pred_flat,average='macro'), \
        recall_score(labels_flat,pred_flat,average="macro"), \
        f1_score(labels_flat, pred_flat,average="macro")
