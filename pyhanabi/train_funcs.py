import os

import torch
import torch.nn.functional as F
import random

def batch(batch_size, shuffle, train_x, train_y, kth_examp):
    batch_x = torch.zeros(batch_size, 80, 660)
    batch_y = torch.zeros(batch_size, 5, 25)
    for j in range(batch_size):
        batch_x[j, 0 : int(kth_examp[shuffle[j]][1]) + 1, :] = torch.tensor(train_x[int(kth_examp[shuffle[j]][0]), 0 : int(kth_examp[shuffle[j]][1]) + 1, :])
        batch_y[j, :, :]                 = torch.tensor(train_y[int(kth_examp[shuffle[j]][0]), int(kth_examp[shuffle[j]][1]), :, :])
    return batch_x.float(), batch_y.float()

def rand_batch(batch_size, num_test, train_x_test, train_y_test, kth_examp_test):
    examps = random.sample(range(0, num_test), batch_size)
    batch_x = torch.zeros(batch_size, 80, 660)
    batch_y = torch.zeros(batch_size, 5, 25)
    for j in range(batch_size):
        batch_x[j, 0 : int(kth_examp_test[examps[j]][1]) + 1, :] = torch.tensor(train_x_test[int(kth_examp_test[examps[j]][0]), 0 : int(kth_examp_test[examps[j]][1]) + 1, :])
        batch_y[j, :, :]                 = torch.tensor(train_y_test[int(kth_examp_test[examps[j]][0]), int(kth_examp_test[examps[j]][1]), :, :])
    return batch_x.float(), batch_y.float()

def get_loss(model, batch_size, src, trg, device, kth_input):
    # input to transformer, conditioning on cards before

    trg_input = trg.detach().clone().to(device)
    trg_input[:, kth_input:, :] *= 0
    
    # the cards we are trying to predict

    tgets = torch.zeros(batch_size, device=device, dtype=torch.long)
    targets = trg[:,kth_input,:].contiguous().view(-1,25)
    for j in range(targets.size(0)):
        if 1 not in targets[j,:]:
            tgets[j] = -420
        else:
            tgets[j] = list(targets[j,:]).index(1)

    preds = torch.zeros(batch_size, 1, 25, device=device)
    preds = model(src, trg_input, None, None)

    loss = F.cross_entropy(preds.contiguous().view(-1,25), 
                            tgets, ignore_index = -420)
    
    return loss

def get_loss_wo_groundtruth(batch_size, preds, trg, device, kth_input):
    # input to transformer, conditioning on cards before
    # the cards we are trying to predict

    tgets = torch.zeros(batch_size, device=device, dtype=torch.long)
    targets = trg[:,kth_input,:].contiguous().view(-1,25)
    for j in range(targets.size(0)):
        if 1 not in targets[j,:]:
            tgets[j] = -420
        else:
            tgets[j] = list(targets[j,:]).index(1)

    loss = F.cross_entropy(preds.contiguous().view(-1,25), 
                            tgets, ignore_index = -420)
    
    return loss

def get_loss_online(batch_size, preds, trg, device, kth_input):
    # input to transformer, conditioning on cards before
    # the cards we are trying to predict

    tgets = torch.zeros(batch_size, dtype=torch.long)
    targets = trg[:,kth_input,:].contiguous().view(-1,26)
    for j in range(targets.size(0)):
        # if 1 not in targets[j,:]:
        #     tgets[j] = -420
        # else:
        
        tgets[j] = list(targets[j,:]).index(1)

    loss = F.cross_entropy(preds.contiguous().view(-1,26), 
                            tgets.to(device))

    return loss