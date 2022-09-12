
import os
import datetime
import logging

import pandas as pd
import csv
import torch



os.environ["CUDA_VISIBLE_DEVICES"]='7'

logging.basicConfig(filename='/home/qd/code/noise-learning/preprocess/mylog',format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d%I:%M:%S %p')

logging.info("use gpu"+str(os.environ["CUDA_VISIBLE_DEVICES"]))


data_path = '/data1/qd/noise_master/dbpedia_csv/small_train.csv'
aug_data_path = '/data1/qd/noise_master/dbpedia_csv/aug_train.csv'
aug_text = '//data1/qd/noise_master/dbpedia_csv/aug.txt'

def process_csv(data_path):
    file = pd.read_csv(data_path,header=None)
    lst = file.values.tolist()
    for row in lst:
        del row[1]
    # ["labels","texts"]
    return lst

def aug_data(data,t=0):
    # Aug train data by back translation of German and Ru
    # Load Transformer model trained on WMT'19 data:
    en2de = torch.hub.load(repo_or_dir='/data1/qd/noise_master/fairseq', \
        model='transformer.wmt19.en-de.single_model', source='local',tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load(repo_or_dir='/data1/qd/noise_master/fairseq', \
        model='transformer.wmt19.de-en.single_model', source='local',tokenizer='moses', bpe='fastbpe')   
    en2ru = torch.hub.load(repo_or_dir='/data1/qd/noise_master/fairseq', \
        model='transformer.wmt19.en-ru.single_model', source='local',tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load(repo_or_dir='/data1/qd/noise_master/fairseq', \
            model='transformer.wmt19.ru-en.single_model', source='local',tokenizer='moses', bpe='fastbpe')

    # en2de=torch.nn.DataParallel(en2de).cuda()
    # de2en=torch.nn.DataParallel(de2en).cuda()
    # en2ru=torch.nn.DataParallel(en2ru).cuda()
    # ru2en=torch.nn.DataParallel(ru2en).cuda()

    en2de.cuda()
    de2en.cuda()
    en2ru.cuda()
    ru2en.cuda()

    # trans_dict_de = {}
    # trans_dict_ru = {}


    for row in data[t:]:
        text=row[1]

        aug1 = de2en.translate(en2de.translate(
            text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        aug2 = ru2en.translate(en2ru.translate(
            text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        row.append(aug1)
        row.append(aug2) # ["labels","texts","de_aug","ru_aug"]

        with open(aug_text,'a',encoding='utf-8') as f:
            tmp = [str(i) for i in row]
            f.write('@@##$$'.join(tmp))
            f.write('\n')

        if t%50==0:
            logging.info(str(t)+"  "+str(len(data)))
        t+=1
    
    return data


# with open(aug_text,'r',encoding='utf-8') as f:
#     aug_data0 = f.readlines()
#     t=len(aug_data0)

data=process_csv(data_path)
aug_data1=aug_data(data,t=0)

f = open(aug_data_path,'w')
writer = csv.writer(f)
writer.writerows(aug_data1)
f.close()