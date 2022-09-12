import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from utils.common import *


class MyDataset(Dataset): 
    def __init__(self, args, mode, dataset, pred=[], probability=[], log=''): 

        self.args = args
        
        self.label = dataset.label
        self.input = dataset.text
        self.cl = torch.tensor(np.array(self.input[:,0],dtype='int64'))

        self.mode = mode     
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_model_dir + self.args.bert_type)

        if self.mode == "labeled":
            pred_idx = pred.nonzero()[0]
            self.input = [self.input[i] for i in pred_idx]
            self.label = self.label[pred_idx]
            self.cl = self.cl[pred_idx]
            self.prob = [probability[i] for i in pred_idx]
            self.pred_idx = pred_idx

        elif self.mode == "unlabeled":
            pred_idx = (1-pred).nonzero()[0]
            self.input = [self.input[i] for i in pred_idx]
            self.label = self.label[pred_idx]
            self.cl = self.cl[pred_idx]
            self.pred_idx = pred_idx
                                          
    def __len__(self):
        return len(self.input)

    def get_tokenized(self, text):

        tokens = self.tokenizer(text,padding='max_length',truncation=True,max_length=self.args.sentence_len,return_tensors='pt')

        for item in tokens:
            tokens[item] = tokens[item].squeeze()
        
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

    def __getitem__(self, index):
        
        text = self.input[index][-1]

        input_id, att_mask = self.get_tokenized(text)

        if self.args.train_aug is False:
            if self.mode=='labeled':
                return input_id, att_mask, self.label[index], self.cl[index], self.prob[index] , self.pred_idx[index]
            elif self.mode=='unlabeled':
                return input_id, att_mask, self.cl[index], self.pred_idx[index]
            elif self.mode=='all':
                return input_id, att_mask, self.label[index], index
        else:
            if self.mode=='labeled':
                token_aug, att_mask1 = self.get_tokenized(self.input[index][2])
                return (input_id, token_aug), (att_mask, att_mask1), (self.label[index], self.label[index]), self.prob[index]
            elif self.mode=='unlabeled':
                token_aug1, att_mask1 = self.get_tokenized(self.input[index][2])
                token_aug2, att_mask2 = self.get_tokenized(self.input[index][3])
                return (token_aug1, token_aug2, input_id), (att_mask1, att_mask2, att_mask)
            elif self.mode=='all':
                return input_id, att_mask, self.label[index], index


class MyDataloader():  
    def __init__(self,args,dataset):
        self.dataset = dataset
        self.args = args
        
        self.batch_size = args.batch_size
        
        self.num_workers = 2

    def run(self,mode,pred=[],prob=[]):

        if mode == "all":
            all_dataset = MyDataset(
                args=self.args,
                dataset=self.dataset, 
                mode="all")
                
            all_loader = DataLoader(
                dataset = all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=5)          
            return all_loader   

        if mode == "train":

            batch_size_m = self.args.batch_size_m

            labeled_dataset = MyDataset(
                args=self.args,
                dataset=self.dataset, 
                mode="labeled", pred=pred,
                probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=batch_size_m,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = MyDataset(
                args=self.args,
                dataset=self.dataset, 
                mode="unlabeled", pred=pred)              
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=batch_size_m,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
            

    def count_label(self):

        label_idx=[]

        labels=self.dataset.label

        for label_id in range(self.args.num_class):
            idx=(labels==label_id)
            label_idx.append(idx)

        return label_idx

    def relabel(self,pred,index):
        mx = torch.max(torch.softmax(pred,dim=-1),dim=-1)[0]
        new_label = torch.argmax(pred,dim=-1)
        for i in range(new_label.size(0)):
            if mx[i] > self.args.p:
                
                # torch.tensor(np.array(self.dataset.text[:,0],dtype='int64'))[index]
                self.dataset.label[index[i]] = new_label[i]

    def relabel2(self,prob,index,t_threshold):

        ori_label = self.dataset.label[index]
        pred_label = torch.argmax(prob,dim=-1)

        for i in range(index.size):
            if prob[i][pred_label[i]] > t_threshold[pred_label[i]] and prob[i][ori_label[i]] < t_threshold[ori_label[i]]:
                # re = (self.dataset.label[index] != pred_label)
                # tr = (pred_label == torch.tensor(np.array(self.dataset.text[:,0],dtype='int64'))[index])
                # import pdb
                # pdb.set_trace()
                # torch.tensor(np.array(self.dataset.text[:,0],dtype='int64'))[index]
                self.dataset.label[index[i]] = pred_label[i]

