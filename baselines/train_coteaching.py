import argparse
import datetime
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

from model.bert_classify import Bert4Classify
from preprocess.dataloader import MyDataloader
from preprocess.read_data import *
from utils.common import get_co_progressbar
from utils.loss import loss_coteaching
from utils.metric import metric

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()

## basic configuration
parser.add_argument('--save_model',action='store_true', default=False)
parser.add_argument('--save_model_dir', default='./save_model/')
parser.add_argument('--pretrain_model_dir', default='./pre_train_models')
parser.add_argument('--train_path', default='./data/train.csv')
parser.add_argument('--test_path', default='./data/test.csv')
parser.add_argument('--noise_ratio',type=float,default=0.0)
parser.add_argument('--noise_type',type=str,default="sym")
parser.add_argument('--fix_data',type=str,default='1')
parser.add_argument('--show_bar',action='store_true', default=False)
parser.add_argument('--seed',type=int,default=128)

## args for train
parser.add_argument('--epoch',type=int,default=6)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--sentence_len',type=int,default=256)
parser.add_argument('--num_class',type=int,default=10)
parser.add_argument('--learning_rate',type=float,default=1e-5)
parser.add_argument('--dropout_rate',type=float,default=0.1)

## args for model
parser.add_argument('--train_aug', type=bool, default=False,
    help='whether to use augement data')
parser.add_argument('--bert_type',type=str,default='bert-base-uncased')
parser.add_argument("--mix_option",type=int,default=0,
    help='mix option for bert , 0: base bert model from huggingface; 1: mix bert')

parser.add_argument('--forget_rate',type=float,
    help='forget rate for each batch')
parser.add_argument('--epoch_decay_start',type=int, default=6,
    help='epoch decay starting for learning rate')
parser.add_argument('--num_gradual',type=int,default=4,
    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent',type=float,default=1,
    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')


args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print_args(args)
setup_seed(args.seed)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"use gpu device =",os.environ["CUDA_VISIBLE_DEVICES"])


EPOCH = args.epoch
BATCH_SIZE = args.batch_size

if args.forget_rate is None:
    forget_rate=args.noise_ratio
else:
    forget_rate=args.forget_rate

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
learning_rate=args.learning_rate

alpha_plan = [learning_rate] * EPOCH
beta1_plan = [mom1] * EPOCH
for i in range(args.epoch_decay_start, EPOCH):
    alpha_plan[i] = float(EPOCH - i) / (EPOCH - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule

rate_schedule = np.ones(EPOCH)*forget_rate
rate_schedule[:args.num_gradual] = \
    np.linspace(0, forget_rate**args.exponent, args.num_gradual)

# Train the Model
def train(args, train_data, epoch, model1, optimizer1, model2, optimizer2):
    
    model1.train()
    model2.train()

    train_loss1 = 0.0
    train_acc1 = 0.0
    train_recall1 = 0.0

    train_loss2 = 0.0
    train_acc2 = 0.0
    train_recall2 = 0.0

    bar = None
    if args.show_bar:
        bar = get_co_progressbar(epoch+1, EPOCH, len(train_data), 'train ')

    for i, data in enumerate(train_data):

        input_ids, attention_mask, labels, _ = [Variable(elem.cuda()) for elem in data]
        
        # Forward + Backward + Optimize
        logits1 = model1(input_ids, attention_mask)
        logits2 = model2(input_ids, attention_mask)
        loss1, loss2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])

        out1 = F.log_softmax(logits1, dim=1)
        out2 = F.log_softmax(logits2, dim=1)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        train_loss1 += loss1.item()
        train_loss2 += loss2.item()

        out1 = out1.cpu().detach().numpy()
        out2 = out2.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        train_metric1 = metric(out1,labels)
        train_acc1 += train_metric1[0]
        train_recall1 += train_metric1[2]

        train_metric2 = metric(out2,labels)
        train_acc2 += train_metric2[0]
        train_recall2 += train_metric2[2]

        if args.show_bar:
                bar.dynamic_messages.loss1 = train_loss1 / (i + 1)
                bar.dynamic_messages.acc1 = train_acc1 / (i + 1)
                bar.dynamic_messages.loss2 = train_loss2 / (i + 1)
                bar.dynamic_messages.acc2 = train_acc2 / (i + 1)
                bar.update(i + 1)
    
    if bar:
        bar.finish()

    train_loss1 /= i+1
    train_loss2 /= i+1
    train_acc1 /= i+1
    train_acc2 /= i+1    

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
    'Train Epoch [%d/%d]: Loss1: %.4F, Accuracy1: %.4f, Loss2: %.4f Accuracy2: %.4f, ' \
        %(epoch+1, EPOCH, train_loss1, train_acc1, train_loss2, train_acc2))

    return train_acc1, train_acc2

# Evaluate the Model
def evaluate(valid_data, epoch, model1, model2, mode='Valid'):

    model1.eval()    # Change model to 'eval' mode.
    model2.eval()

    val_loss1 = 0.0
    val_acc1 = 0.0
    val_recall1 = 0.0

    val_loss2 = 0.0
    val_acc2 = 0.0
    val_recall2 = 0.0

    for j, data in enumerate(valid_data):

        input_ids, attention_mask, labels, _ = [Variable(elem.cuda()) for elem in data]

        attention_mask = Variable(attention_mask.cuda())

        with torch.no_grad():
            logits1 = model1(input_ids, attention_mask)
            logits2 = model2(input_ids, attention_mask)
            out1 = F.log_softmax(logits1, dim=1)
            out2 = F.log_softmax(logits2, dim=1)
            loss1 = F.cross_entropy(logits1, labels, reduce=False)
            loss2 = F.cross_entropy(logits2, labels, reduce=False)

            out1 = out1.cpu().detach().numpy()
            out2 = out2.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            metric1 = metric(out1,labels)
            metric2 = metric(out2,labels)

            val_loss1 += loss1.mean()
            val_acc1 += metric1[0] * labels.size
            val_recall1 += metric1[2] * labels.size

            val_loss2 += loss2.mean()
            val_acc2 += metric2[0] * labels.size
            val_recall2 += metric2[2] * labels.size

    val_loss1 /= j+1
    val_acc1 /= len(valid_data.dataset)
    val_recall1 /= len(valid_data.dataset)
    val_loss2 /= j+1
    val_acc2 /= len(valid_data.dataset)
    val_recall2 /= len(valid_data.dataset)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), mode,
    'Epoch [%d/%d]: Loss1: %.4F, Accuracy1: %.4f, Loss2: %.4f, Accuracy2: %.4f' \
        %(epoch+1, EPOCH, val_loss1, val_acc1, val_loss2, val_acc2))

    return val_acc1, val_acc2


def main():

    # Data Loader 
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"load data from",args.train_path)

    train_data, valid_data  = process_csv(args, args.train_path)
    test_data = process_test_csv(args, args.test_path)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train data %d , valid data %d , test data %d " \
        %(len(train_data),len(valid_data),len(test_data)))

    train_loader = MyDataloader(args,train_data).run("all")
    #valid_loader = MyDataloader(args,valid_data).run("all")
    test_loader = MyDataloader(args,test_data).run("all")

    # Define models

    model1 = Bert4Classify(args)
    model1 = torch.nn.DataParallel(model1).cuda()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
    
    model2 = Bert4Classify(args)
    model2 = torch.nn.DataParallel(model2).cuda()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)

    test_best_l = []
    test_last_l = []

    test_best=0.0

    # training
    for epoch in range(0, EPOCH):

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)
        
        train(args, train_loader, epoch, model1, optimizer1, model2, optimizer2)
        test_acc1, test_acc2 =evaluate(test_loader, epoch, model1, model2, 'Test')

        test_last = max(test_acc1, test_acc2)
        test_best = max(test_best, test_last)

        test_best_l.append(test_best)
        test_last_l.append(test_last)

    print('test_best',test_best_l)
    print('test_last',test_last_l)
    
    print("Test best %f , last %f"%(test_best, test_last))


if __name__=='__main__':
    main()
