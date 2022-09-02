import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


logger = logging.getLogger(__name__)

NUM_CLASSES = {
    'trec': 6,
    'imdb': 2,
    'agnews': 4
}

class DataToDataset(Dataset):
    def __init__(self, data):
        self.labels, self.texts = data.values[:, 0], data.values[:, 1]
        
    def __len__(self):
        return len(self.labels)
            
    def __getitem__(self,index):
        return self.texts[index], self.labels[index]
    

def load_dataset(data_path, dataset_name):
    extension = data_path.split(".")[-1]
    assert extension == 'csv'
    
    data = pd.read_csv(data_path, header=None)
    
    if dataset_name in NUM_CLASSES:
        num_classes = NUM_CLASSES[dataset_name]
    else:
        num_classes = max(data.values[:, 0]) + 1
        
    logger.info('num_classes is %d', num_classes)
    return DataToDataset(data), num_classes
    

class SelfMixDataset(Dataset):
    def __init__(self, data_args, dataset, tokenizer, mode, pred=[], probability=[]): 

        self.data_args = data_args
        self.labels = dataset.labels
        self.inputs = dataset.texts

        self.mode = mode
        self.tokenizer = tokenizer

        if self.mode == "labeled":
            pred_idx = pred.nonzero()[0]
            self.inputs = [self.inputs[idx] for idx in pred_idx]
            self.labels = self.labels[pred_idx]
            self.prob = [probability[idx] for idx in pred_idx]
            self.pred_idx = pred_idx

        elif self.mode == "unlabeled":
            pred_idx = (1 - pred).nonzero()[0]
            self.inputs = [self.inputs[idx] for idx in pred_idx]
            self.labels = self.labels[pred_idx]
            self.pred_idx = pred_idx
                                          
    def __len__(self):
        return len(self.inputs)

    def get_tokenized(self, text):
        tokens = self.tokenizer(text, padding='max_length', truncation=True, 
                                max_length=self.data_args.max_sentence_len, return_tensors='pt')

        for item in tokens:
            tokens[item] = tokens[item].squeeze()
        
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

    def __getitem__(self, index):
        text = self.inputs[index]
        input_id, att_mask = self.get_tokenized(text)
        if self.mode == 'labeled':
            return input_id, att_mask, self.labels[index], self.prob[index], self.pred_idx[index]
        elif self.mode == 'unlabeled':
            return input_id, att_mask, self.pred_idx[index]
        elif self.mode == 'all':
            return input_id, att_mask, self.labels[index], index


class SelfMixData:
    def __init__(self, data_args, datasets, tokenizer):
        self.data_args = data_args
        self.datasets = datasets
        self.tokenizer = tokenizer
    
    def run(self, mode, pred=[], prob=[]):
        if mode == "all":
            all_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="all")
                
            all_loader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.data_args.batch_size,
                shuffle=True,
                num_workers=2)          
            return all_loader

        if mode == "train":
            labeled_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="labeled", 
                pred=pred, probability=prob)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.data_args.batch_size_mix,
                shuffle=True,
                num_workers=2)   
            
            unlabeled_dataset = SelfMixDataset(
                data_args=self.data_args,
                dataset=self.datasets,
                tokenizer=self.tokenizer,
                mode="unlabeled", 
                pred=pred)              
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.data_args.batch_size_mix,
                shuffle=True,
                num_workers=2)     
            return labeled_trainloader, unlabeled_trainloader
