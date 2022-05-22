import argparse
import datetime
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable

from model import Bert4Classify
from dataloader import MyDataloader
from read_data import *
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

### basic configuration
parser.add_argument('--results_dir', default='./save_model/')
parser.add_argument('--train_path', default='./data/train.csv')
parser.add_argument('--test_path', default='./data/test.csv')
parser.add_argument('--noise_ratio',type=float,default=0.0)
parser.add_argument('--noise_type',type=str,default="asym",
                    help='sym, asym or idn')
parser.add_argument('--show_bar',action='store_true', default=False)
parser.add_argument('--seed',type=int,default=32)

### args for train
parser.add_argument('--num_class',type=int)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--batch_size_m',type=int,default=16,
                    help='batch size for mix train')
parser.add_argument('--sentence_len',type=int,default=256)
parser.add_argument('--bert_type', type=str, default='bert-base-uncased')

parser.add_argument('--warmup_epoch', type=int, default=2)
parser.add_argument('--warmup_samples', type=int, default=float('inf'),
                    help='max samples for each warmup epoch')
parser.add_argument('--epoch',type=int,default=4,
                    help='Mix-up train epoch')

parser.add_argument('--learning_rate',type=float,default=1e-5)
parser.add_argument('--dropout_rate',type=float,default=0.1)
parser.add_argument('--accumulation_steps',type=int, default=1,
                    help='gradient accumulation step')

parser.add_argument('--p_threshold', type=float, default=0.5,
                    help='clean probability threshold')
parser.add_argument('--T', type=float, default=0.5,
                    help='temperature for sharpen function')
parser.add_argument('--alpha', type=float, default=0.75,
                    help='alpha for beta distribution') 
parser.add_argument('--lambda_r', type=float, default=0.3,
                    help='weight for R-drop Loss')                 
parser.add_argument('--lambda_p', type=float, default=0.2,
                    help='weight for Pseudo Loss')
parser.add_argument('--class_regularization',action='store_true', default=False)

args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(args.seed)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"use gpu device =",os.environ["CUDA_VISIBLE_DEVICES"])

EPOCH = args.epoch

def main():
    # Read dataset and build dataloaders
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"load data from",args.train_path, args.test_path)

    train_data, valid_data  = process_csv(args, args.train_path)
    test_data = process_test_csv(args, args.test_path)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"train data %d , valid data %d , test data %d " 
    %(len(train_data),len(valid_data),len(test_data)))

    print_args(args)

    train_data_loader = MyDataloader(args,train_data)
    # valid_data_loader = MyDataloader(args,valid_data)
    test_data_loader = MyDataloader(args,test_data)

    train_data = train_data_loader.run("all")
    # valid_data = valid_data_loader.run("all")
    test_data = test_data_loader.run("all")

    criterion = nn.CrossEntropyLoss()
    CE = nn.CrossEntropyLoss(reduction='none')
    
    model = Bert4Classify(args).cuda()
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

    print('\n====================================== warmup for model ======================================\n')
    
    warmup(args.warmup_epoch, model, optimizer, train_data, test_data, criterion)

    test_best = 0.0
    test_best_l = []
    test_last_l = []

    print('\n====================================== start mix-up train ======================================\n')

    for epoch in range(0, EPOCH):          

        prob, class_prob = eval_train(model,train_data,CE)
        pred = (prob > args.p_threshold) 

        labeled_trainloader, unlabeled_trainloader = train_data_loader.run('train',pred,prob) 
        count_noise(labeled_trainloader, unlabeled_trainloader) 

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch)

        (test_loss, test_last, _) = evaluate(model, test_data, criterion)
        if test_last > test_best:
            test_best=test_last
            torch.save(model.state_dict(), args.results_dir+'/best.ckpt')

        test_best_l.append(test_best)
        test_last_l.append(test_last)
 
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Test Epoch [%d/%d]: Loss1: %.4F, Accuracy1: %.4f ' \
            %(epoch+1, EPOCH, test_loss, test_last))

    torch.save(model.state_dict(), args.results_dir+'/last.ckpt')

    print('test_best',test_best_l)
    print('test_last',test_last_l)
    print("Test best %f , last %f"%(test_best, test_last))


def warmup(epochs,model,optimizer,dataloader,valid_data,Loss):
    
    bar = None
    all_steps = 0
    
    for epoch in range(1,epochs+1):
        model.train()
        if args.show_bar:
            bar = get_progressbar(epoch, epochs, len(dataloader), 'WarmUp ')
        train_loss = 0.0
        train_acc = 0.0
        train_recall = 0.0
        for i, data in enumerate(dataloader):  
            ids, att, labels, _  = [Variable(elem.cuda()) for elem in data]

            logits = model(ids, att, is_training=True)

            loss = Loss(logits, labels)

            train_loss += loss.mean().item()
            
            out = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            train_metric = metric(out,labels)
            train_acc += train_metric[0]
            train_recall += train_metric[2]

            if args.show_bar:
                bar.dynamic_messages.loss = train_loss / (i + 1)
                bar.dynamic_messages.acc = train_acc / (i + 1)
                bar.dynamic_messages.recall = train_recall / (i + 1)
                bar.update(i + 1)

            loss = loss/args.accumulation_steps
            loss.backward()

            if((i+1)%args.accumulation_steps)==0 or i+1 == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            batch_size = ids.shape[0]
            all_steps += batch_size
            if all_steps >= args.warmup_samples:
                print(f"stop in {all_steps}")
                break
            
        if bar:
            bar.finish()
        
        # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # 'Warmup Epoch [%d/%d]: Loss: %.4F, Accuracy: %.4f ' \
        #     %(epoch, epochs, train_loss/len(dataloader), train_acc/len(dataloader)))

        if valid_data:
            eval_res = evaluate(model, valid_data, Loss)

            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Warmup Test Epoch [%d/%d]: Loss: %.4F, Accuracy: %.4f ' \
                %(epoch, epochs, eval_res[0], eval_res[1]))
        
    return 


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch):
    

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    val_iteration = len(labeled_trainloader)
    
    model.train()

    bar = None

    if args.show_bar:
        bar = get_progressbar(epoch+1, EPOCH, val_iteration, 'Train')

    for batch_idx in range(val_iteration):

        try:
            inputs_x, inputs_x_att, targets_x, _ , w_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x_att, targets_x, _ , w_x, _ = labeled_train_iter.next()

        try:
            inputs_u, att_u, _, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, att_u, _, _ = unlabeled_train_iter.next()
       
        
        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_u.size(0)
            
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1, 1), 1)

        inputs_x, inputs_x_att, targets_x = inputs_x.cuda(), inputs_x_att.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u, att_u = inputs_u.cuda(), att_u.cuda()
        
        w_x = w_x.view(-1,1).type(torch.FloatTensor).cuda() 
        
        with torch.no_grad():

            model.eval()

            # Predict labels for unlabeled data.

            out_u = model(inputs_u, att_u, is_training=False)
            
            p=torch.softmax(out_u, dim=1) 

            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        model.train()

        # EmbMix

        inputs_x = model.module.emb(inputs_x, inputs_x_att)
        inputs_u2 = model.module.emb(inputs_u, att_u)
        inputs_u = model.module.emb(inputs_u, att_u)

        all_inputs = torch.cat(
            [inputs_x, inputs_u], dim=0)
        all_targets = torch.cat(
            [targets_x, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        mixed_input = l * input_a + (1 - l) * input_b 
        mixed_target = l * target_a + (1 - l) * target_b

        logits = model.module.classify(mixed_input,is_training=True)
        
        logits_u = model.module.classify(inputs_u,is_training=True)
        logits_u2 = model.module.classify(inputs_u2,is_training=True)


        loss0 = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))            

        pse_loss = -torch.mean(F.log_softmax(logits_u, dim=1).min(dim=1)[0]) * 0.5 - torch.mean(F.log_softmax(logits_u2, dim=1).min(dim=1)[0]) * 0.5

        kl_loss = compute_kl_loss(logits_u,logits_u2)
        
        loss = loss0 + kl_loss * args.lambda_r + pse_loss * args.lambda_p

        if batch_idx==val_iteration-1:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Loss_Mix: %.4F, Loss_R: %.4f, Loss_P: %.4f, Loss: %.4f ' \
                %(loss0, kl_loss, pse_loss, loss))
        
        loss = loss/args.accumulation_steps
        loss.backward()

        if((batch_idx+1)%args.accumulation_steps)==0 or batch_idx+1 == val_iteration:
            optimizer.step()
            optimizer.zero_grad()

        if args.show_bar:
            bar.dynamic_messages.loss = loss 
            bar.update(batch_idx+1)

    if bar:
        bar.finish()


def eval_train(model,dataloader,Loss):    
    '''
    Sample selection
    '''
    model.eval()
    losses = torch.zeros(len(dataloader.dataset.label))
    class_prod = torch.zeros(len(dataloader.dataset.label),args.num_class)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            ids, att, label, index = [Variable(elem.cuda()) for elem in data] 
            outputs = model(ids, att, is_training=False) 
            pred = torch.softmax(outputs,dim=1)
            loss = Loss(pred, label)

            for b in range(ids.size(0)):
                losses[index[b]]=loss[b]
                class_prod[index[b]]=pred[b]

    if args.class_regularization:
        for now_class in range(args.num_class):
            indices = np.where(dataloader.dataset.label == now_class)[0]
            losses[indices] = (losses[indices] - losses[indices].mean()) / losses[indices].var()
    else:
        losses = (losses-losses.min())/(losses.max()-losses.min())

    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    losses = losses.reshape(-1, 1)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]

    return prob,class_prod


def evaluate(model,valid_data,Loss):    
    
    val_loss = 0.0
    val_acc = 0.0
    val_recall = 0.0
    model.eval()

    for j, batch in enumerate(valid_data):
        val_input_ids, val_att, val_labels, _ = [Variable(elem.cuda()) for elem in batch]
        with torch.no_grad():
            pred = model(val_input_ids, val_att, is_training=False)
            val_loss += Loss(pred, val_labels).mean()
    
            pred = pred.cpu().detach().cpu().numpy()
            val_labels = val_labels.cpu().detach().cpu().numpy()
            val_metric = metric(pred, val_labels)
            val_acc += val_metric[0]* val_labels.size
            val_recall += val_metric[2] * val_labels.size
        
    val_loss /= len(valid_data)
    val_acc /= len(valid_data.dataset)
    val_recall /= len(valid_data.dataset)

    return val_loss, val_acc, val_recall



if __name__ == '__main__':
    
    main()

    
