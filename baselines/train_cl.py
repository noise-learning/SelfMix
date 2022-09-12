import argparse
import configparser
from copy import deepcopy
import os
import time
import warnings
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from model.bert_classify import Bert4Classify
from preprocess.dataloader import *
from preprocess.read_data import *
from utils.common import *
from utils.metric import *
from utils.clean_data import LearningWithNoisyLabels
from sklearn.base import BaseEstimator


os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# basic configuration
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
parser.add_argument('--train_aug',type=bool, default=False,
    help='whether to use augement data')
parser.add_argument('--bert_type',type=str,default='bert-base-uncased')
parser.add_argument("--mix_option",type=int,default=0,
    help='mix option for bert , 0: base bert model from huggingface; 1: mix bert')

## args for confident learning
parser.add_argument('--cv_epoch',type=int,default=3)
parser.add_argument('--prune_method',type=str,default='prune_by_noise_rate')

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


tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_dir + args.bert_type)

def get_tokenized(text):
    tokens = tokenizer(text,padding='max_length',truncation=True,max_length=args.sentence_len,return_tensors='pt')
    input_id, attmsk_id = tokens['input_ids'], tokens['attention_mask']
    return np.concatenate((input_id, attmsk_id), axis=1)


train_data, valid_data  = process_csv(args, args.train_path)
X_train, y_train = get_tokenized(list(train_data.text[:, -1])), train_data.label.numpy()
# X_valid, y_valid = get_tokenized(list(valid_data.text[:, 1])), valid_data.label.numpy()
test_data = process_test_csv(args, args.test_path)
X_test, y_test = get_tokenized(list(test_data.text[:, -1])), test_data.label.numpy()

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train data %d , valid data %d , test data %d " 
%(len(train_data),len(valid_data),len(test_data)))


class TrainDataset(Dataset):
    def __init__(self, text_att, label) -> None:
        super().__init__()
        self.text, self.attmsk = np.hsplit(text_att, 2)
        self.label = label
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.text[index], self.attmsk[index], self.label[index]


class PredDataset(Dataset):
    def __init__(self, text_att) -> None:
        super().__init__()
        self.text, self.attmsk = np.hsplit(text_att, 2)
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.attmsk[index]


class MyModel(BaseEstimator):
    def __init__(
        self,
        model,
        optimizer,
        test_x=None,
        test_y=None,
        batch_size=32,
        cv_epoch=3,
        retrain_epoch=6,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.cv_epoch = cv_epoch
        self.retrain_epoch = retrain_epoch
        self.epoch = 0
        self.test_x = test_x
        self.test_y = test_y
        self.best_acc = 0
        self.mode = 'filter noisy labels'
        self.retrain_last_acc = []
        self.retrain_best_acc = []

    def fit(self, X, y, sample_weight=None):
        print(f"Train data size = {X.shape[0]}")
        train_set = TrainDataset(X, y)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=5
        )
        if self.mode == "filter noisy labels":
            self.epoch = self.cv_epoch
        else:
            self.epoch = self.retrain_epoch
        for epoch in range(1, self.epoch + 1):
            train_acc, train_recall = 0, 0
            self.model.train()
            for batch_idx, (data, attmsk, labels) in enumerate(train_loader):
                data, attmsk, labels = Variable(data).cuda(), Variable(attmsk).cuda(), Variable(labels).cuda()
                self.optimizer.zero_grad()
                logits = self.model(data, attmsk, is_training=True)
                out = F.log_softmax(logits)
                loss = F.cross_entropy(logits, labels)
                
                out = out.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                train_metric = metric(out,labels)
                train_acc += train_metric[0] * data.shape[0]
                train_recall += train_metric[2] * data.shape[0]

                loss.mean().backward()
                self.optimizer.step()
            
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train %d/%d Acc:%f, Recall:%f" \
            %(epoch, self.epoch, train_acc / X.shape[0], train_recall / X.shape[0]))
            if self.mode == "retrain" and self.test_x is not None and self.test_y is not None:
                now_metric = self.score(self.test_x, self.test_y)
                self.best_acc = max(self.best_acc, now_metric[0])
                self.retrain_last_acc.append(now_metric[0])
                self.retrain_best_acc.append(self.best_acc)
                print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"test %d/%d Acc:%f, Recall:%f" \
                %(epoch, self.epoch, now_metric[0], now_metric[2]))
        if self.mode == 'retrain':
            return self.retrain_best_acc, self.retrain_last_acc


    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


    def predict_proba(self, X):
        pred_set = PredDataset(X)
        pred_loader = DataLoader(
            dataset=pred_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=5
        )
        self.model.eval()

        outputs = []
        for text, attmsk in pred_loader:
            with torch.no_grad():
                text = Variable(text).cuda()
                attmsk = Variable(attmsk).cuda()
                output = self.model(text, attmsk, is_training=False)
                output = torch.softmax(output, dim=1)
                outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        out = outputs.cpu().numpy()
        return out


    def score(self, X, y, sample_weight=None):
        pred = self.predict_proba(X)
        now_metric = metric(pred, y)
        return now_metric


mymodel=Bert4Classify(args)
mymodel=torch.nn.DataParallel(mymodel).cuda()

optimizer=optim.Adam(mymodel.parameters(),lr=args.learning_rate)

clf = MyModel(
    model=mymodel,
    optimizer=optimizer,
    test_x=X_test,
    test_y=y_test,
    batch_size=BATCH_SIZE,
    cv_epoch=args.cv_epoch,
    retrain_epoch=args.epoch
)

clf.mode = 'filter noisy labels'
lnl = LearningWithNoisyLabels(clf=clf, seed=args.seed, prune_method=args.prune_method)
X_clean, y_clean, noise_idx = lnl.fit(X_train, y_train)
clf.mode = 'retrain'
test_best, test_last = clf.fit(X_clean, y_clean)
print(f'test_best {test_best}')
print(f'test_last {test_last}')
print(f'Test best {test_best[-1]}, last {test_last[-1]}')
