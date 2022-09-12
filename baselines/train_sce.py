import argparse
import datetime
import os
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model.bert_classify import Bert4Classify
from preprocess.dataloader import *
from preprocess.read_data import *
from utils.common import *
from utils.metric import *
from utils.loss import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

## basic configuration
parser.add_argument('--save_model',action='store_true', default=True)
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

## args for sce loss
parser.add_argument('--alpha', type=float, default=0.4,
    help='the coefficient of ce loss')
parser.add_argument('--beta', type=float, default=1.0,
    help='the coefficient of rce loss')


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

EPOCH = args.epoch
BATCH_SIZE = args.batch_size

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"use gpu device =",os.environ["CUDA_VISIBLE_DEVICES"])

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"load data from",args.train_path)

train_data, valid_data  = process_csv(args, args.train_path)
test_data = process_test_csv(args, args.test_path)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train data %d , valid data %d , test data %d " \
    %(len(train_data),len(valid_data),len(test_data)))

train_loader = MyDataloader(args,train_data).run("all")
#valid_loader = MyDataloader(args,valid_data).run("all")
test_loader = MyDataloader(args,test_data).run("all")

def train(args, mymodel, optimizer, train_data, valid_data=None, test_data=None):
    
    test_best_l = []
    test_last_l = []
    
    test_best = 0.0

    # evaluate(valid_loader, 0, mymodel, 'valid before train')

    for epoch in range(1, EPOCH + 1):

        train_loss = 0.0
        train_acc = 0.0
        train_recall = 0.0

        mymodel.train()
        bar = None

        if args.show_bar:
            bar = get_progressbar(epoch, EPOCH, len(train_data), 'train')

        for i, data in enumerate(train_data):

            input_ids, attention_mask, labels, _ = [Variable(elem.cuda()) for elem in data]
            optimizer.zero_grad()

            logits = mymodel(input_ids, attention_mask, is_training=True)

            out = F.log_softmax(logits)
            loss = compute_sce_loss(logits, labels, args.alpha, args.beta)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            out = out.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            train_metric = metric(out,labels)
            train_acc += train_metric[0] * labels.size
            train_recall += train_metric[2] * labels.size

            if args.show_bar:
                bar.dynamic_messages.loss = train_loss / (i + 1)
                bar.dynamic_messages.acc = train_acc / (i*args.batch_size + labels.size)
                bar.dynamic_messages.recall = train_recall / (i*args.batch_size + labels.size)
                bar.update(i + 1)
        
        if bar:
            bar.finish()

        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train %d/%d epochs Loss:%f, Acc:%f, Recall:%f" \
        %(epoch, EPOCH, train_loss / (i + 1), train_acc / len(train_data.dataset), train_recall / len(train_data.dataset)))
        
        if valid_data:
            _, val_acc = evaluate(valid_data, epoch, mymodel, 'valid')

        if test_data:
            _, test_last = evaluate(test_data, epoch, mymodel, 'test')
            test_best = max(test_best,test_last)

            test_best_l.append(test_best)
            test_last_l.append(test_last)
        
    return test_best_l, test_last_l

def evaluate(data, epoch, mymodel, mode):
    loss = 0.0
    acc = 0.0
    recall = 0.0
    mymodel.eval()

    for j, batch in enumerate(data):   
        input_ids, attention_mask, labels, _ = [Variable(elem.cuda()) for elem in batch]
     
        with torch.no_grad():
            logits = mymodel(input_ids, attention_mask, is_training=False)
            loss += F.cross_entropy(logits, labels).mean()
            pred = F.log_softmax(logits)
            pred = pred.cpu().detach().cpu().numpy()
            labels = labels.cpu().detach().cpu().numpy()

            metric_ = metric(pred, labels)
            acc += metric_[0] * labels.size
            recall += metric_[2] * labels.size
  
    loss /= len(data)
    acc /= len(data.dataset)
    recall /= len(data.dataset)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), mode, " %d/%d epochs Loss:%f, Acc:%f, Recall:%f" \
    %(epoch, EPOCH, loss , acc , recall))

    return loss, acc

mymodel=Bert4Classify(args)
mymodel=torch.nn.DataParallel(mymodel).cuda()

optimizer=optim.Adam(mymodel.parameters(),lr=args.learning_rate)

test_best_l, test_last_l = train(args, mymodel, optimizer, train_loader, test_data=test_loader)

print('test_best',test_best_l)
print('test_last',test_last_l)
print("Test best %f , last %f"%(test_best_l[-1], test_last_l[-1]))
