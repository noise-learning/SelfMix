import torch
import torch.nn.functional as F


def select_by_confidence0(logits1,logits2):
    return (1 * torch.softmax(logits1, dim=1) + 1 * torch.softmax(logits2, dim=1)) / (2)

def select_by_confidence1(logits1,logits2):
    #整个batch的墒比较 选high entropy
    entropy1 = F.cross_entropy(logits1,F.softmax(logits1))
    entropy2 = F.cross_entropy(logits2,F.softmax(logits2))
    idx = (entropy1 > entropy2)
    w=torch.tensor(idx,dtype=float)
    logits = logits1*w+(1-w)*logits2
    return logits

def select_by_confidence2(logits1,logits2):
    #每个sample的墒比较 选high entropy
    entropy1 = -1*torch.sum(F.log_softmax(logits1)*F.softmax(logits1),dim=-1)
    entropy2 = -1*torch.sum(F.log_softmax(logits2)*F.softmax(logits2),dim=-1)
    idx = (entropy1 > entropy2)
    w=torch.tensor(idx,dtype=float)
    w=w.view(logits1.shape[0],1)
    w.expand(logits1.shape[0],logits1.shape[1])
    logits = logits1*w+(1-w)*logits2
    return logits

def select_by_confidence3(logits1,logits2):
    #整个batch的墒比较 选low entropy
    entropy1 = F.cross_entropy(logits1,F.softmax(logits1))
    entropy2 = F.cross_entropy(logits2,F.softmax(logits2))
    idx = (entropy1 < entropy2)
    w=torch.tensor(idx,dtype=float)
    logits = logits1*w+(1-w)*logits2
    return logits

def select_by_confidence4(logits1,logits2):
    #每个sample的墒比较 选low entropy
    entropy1 = -1*torch.sum(F.log_softmax(logits1)*F.softmax(logits1),dim=-1)
    entropy2 = -1*torch.sum(F.log_softmax(logits2)*F.softmax(logits2),dim=-1)
    idx = (entropy1 < entropy2)
    w=torch.tensor(idx,dtype=float)
    w=w.view(logits1.shape[0],1)
    w.expand(logits1.shape[0],logits1.shape[1])
    logits = logits1*w+(1-w)*logits2
    return logits

def select_by_confidence5(logits1,logits2):
    #entropy loss 
    num_class = logits1.size(1)
    y1 = torch.argmax(logits1,dim=-1)
    hot1 = F.one_hot(y1,num_class)
    y2 = torch.argmax(logits2,dim=-1)
    hot2 = F.one_hot(y2,num_class)
    entropy1 = -1*torch.sum(hot1*F.log_softmax(logits1),dim=-1)
    entropy2 = -1*torch.sum(hot2*F.log_softmax(logits2),dim=-1)
    idx = (entropy1 < entropy2)
    w=torch.tensor(idx,dtype=float)
    w=w.view(logits1.shape[0],1)
    # w.expand(logits1.shape[0],logits1.shape[1])
    # print(w)
    logits = logits1*w+(1-w)*logits2
    return logits

def select_by_confidence6(logits1,logits2):
    #整个batch的墒比较 选high entropy
    logits = (logits1 + logits2) / 2 
    return logits