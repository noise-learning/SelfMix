import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DataToDataset(Dataset):
    def __init__(self,data,label):
        self.text = data["texts"]
        self.label = label
        self.cl = torch.tensor(np.array(self.text[:,0],dtype='int64')) # clean label
        
    def __len__(self):
        return len(self.label)
            
    def __getitem__(self,index):
        return self.text[index], self.label[index]


def read_data(data):

    labels = data["labels"]
    label_id=torch.tensor(labels)
    
    datasets=DataToDataset(data, label_id)

    return datasets

def process_csv(args, data_path, train_ratio=1.0):
    
    file = pd.read_csv(data_path,header=None)

    num = len(file)
    train={'texts':file[:int(train_ratio * num)].values}
    train['labels']=get_label_id(args, file[:int(train_ratio * num)], False)

    if train_ratio<1.0:
        valid={'texts':file[int(train_ratio * num):].values}
        valid['labels']=get_label_id(args, file[int(train_ratio * num):], True)
        
        return read_data(train), read_data(valid)

    return read_data(train), []


def process_test_csv(args, data_path):

    file = pd.read_csv(data_path,header=None)

    data={'texts':file.values}
    data['labels']=get_label_id(args, file, True)

    return read_data(data)

def get_label_id(args, file, clean=True):

    labels = [i[0] for i in file.values]
    num_labels = max(labels)+1
    args.num_class=num_labels

    if not clean:
        labels = flip_label(labels, num_labels, args.noise_type,args.noise_ratio)
    
    return labels

def flip_label(y, n_class, pattern, ratio, one_hot=False):
    ''' 
    Randomly generate noise labels
    y: true label, one hot
    n_class: num classes
    pattern: 'asym' or 'sym'        
    ratio: float, noisy ratio
    '''
    
    if one_hot:
        y = np.argmax(y,axis=1) #[np.where(r==1)[0][0] for r in y]

    for i in range(len(y)):
        if pattern=='sym':
            p1 = ratio/(n_class-1)*np.ones(n_class)
            p1[y[i]] = 1-ratio
            y[i] = np.random.choice(n_class,p=p1)
        elif pattern=='asym':
            if y[i]==5:
                ny=np.random.randint(5)
                for j in range(i+1,len(y)):
                    if y[j]==ny:
                        y[j] = np.random.choice([y[j],5],p=[1-ratio,ratio])
                        break
            elif y[i]==4:
                ny=0
            else:
                ny=y[i]+1
            y[i] = np.random.choice([y[i],ny%n_class],p=[1-ratio,ratio])            
            
    #convert back to one hot
    if one_hot:
        y = np.eye(n_class)[y]

    return y
