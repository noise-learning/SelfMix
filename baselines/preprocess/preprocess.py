import csv
from copy import deepcopy
import random
import numpy as np
import re
import pandas as pd

# label noise
def flip_label(yy, n_class, pattern, ratio, one_hot=False):
    y = deepcopy(yy)
    for i in range(len(y)):
        if pattern=='sym':
            p1 = ratio/(n_class-1)*np.ones(n_class)
            p1[y[i]] = 1-ratio
            y[i] = np.random.choice(n_class,p=p1)
        elif pattern=='asym':
            y[i] = np.random.choice([y[i],(y[i]+1)%n_class],p=[1-ratio,ratio])            
    return y

# preprocess yahoo dataset
def clean_sentence(sent):
    try:
        sent = sent.replace('\n', ' ').replace('\\n', ' ').replace('\\', ' ').replace("\"", ' ')

        sent = re.sub('<[^<]+?>', '', sent)

        return sent.lower()
    except:
        print(sent)
        return ' '

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def mk_data(data_path, res_path, nums, f_noise=False):

    file=pd.read_csv(data_path,header=None)
    data=file.values.tolist()

    num_class=max(file[0])

    class_data=[[] for i in range(num_class)]

    for row in data:
        text=' '.join([str(i) for i in row[1:]])
        text=clean_sentence(text)
        label_id = int(row[0]) -1 

        class_data[label_id].append(text)

    labels=[]
    texts=[]

    for id in range(num_class):
        label=[id]*nums
        text=random.sample(class_data[id],nums)
        labels.extend(label)
        texts.extend(text)

    res_data = pd.DataFrame({'label':labels,'text':texts})

    if f_noise:
        for ratio in [0.1,0.2,0.3,0.4]:
            for noise in ['sym','asym']:
                res_data[str(ratio)+noise]=flip_label(labels,num_class,noise,ratio)
        
        res_data.to_csv(res_path,index=0)
    else:
        res_data.to_csv(res_path,header=None,index=0)

    return res_data

setup_seed(233)




train_path0 = '/data1/qd/noise_master/yahoo/yahoo_answers_csv/train.csv'
train_path1 = '/data1/qd/noise_master/yahoo/yahoo_answers_csv/train_5k.csv'

test_path0 = '/data1/qd/noise_master/yahoo/yahoo_answers_csv/test.csv'
test_path1 = '/data1/qd/noise_master/yahoo/yahoo_answers_csv/test_5k.csv'

mk_data(train_path0,train_path1,500,True)
mk_data(test_path0,test_path1,50,False)

