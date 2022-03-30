import torch
import numpy as np
import torch.nn.functional as F
from train_funcs import get_loss, get_loss_wo_groundtruth, get_loss_online

def get_per_card_data(train_x_test, train_y_test, max_seqs, num_game_steps):
    index = 0
    x_test = torch.zeros(1000, num_game_steps, 660)
    y_test = torch.zeros(1000, num_game_steps, 5, 25)
    for i in range(len(max_seqs)):
        if max_seqs[i] == num_game_steps:
            for j in range(num_game_steps):
                x_test[index,j,:] = torch.tensor(train_x_test[i,j,:])
                y_test[index,j,:] = torch.tensor(train_y_test[i,j,:,:])
            index += 1
            if index == 1000:
                break
    return x_test.float(), y_test.float()

def per_card_scores(x_test, y_test, model, num_game_steps):
    # x_test is 1000 x num_game_steps x encod_dim
    # y_test is 1000 x num_game_steps x 5 x decod_dim

    per_card_scores = np.zeros((num_game_steps,5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_test = torch.zeros((1000, 80, 660), device=device)
    trg_test = torch.zeros((1000, 5, 25), device= device)

    for j in range(5):
        model[j].eval()
    with torch.no_grad():
        for i in range(num_game_steps):
            # At each game step i, we run the model on 1000 examples
            src_test[:, i, :] = x_test[:,i,:]
            trg_test = y_test[:,i,:,:]
            for k in range(1000 // 50):
                for j in range(5):
                    loss_test = get_loss(model[j], 50, src_test[50*k:50*(k+1),:,:], trg_test[50*k:50*(k+1),:,:], device, j)
                    per_card_scores[i,j] += loss_test.item()
            for j in range(5):
                per_card_scores[i,j] /= (1000 // 50)
            
    return per_card_scores

def per_card_scores_wo_groundtruth(x_test, y_test, model, num_game_steps):
    # x_test is 1000 x num_game_steps x encod_dim
    # y_test is 1000 x num_game_steps x 5 x decod_dim

    per_card_scores = np.zeros((num_game_steps,5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_test = torch.zeros((1000, 80, 660), device=device)
    trg_test = torch.zeros((1000, 5, 25), device= device)

    for j in range(5):
        model[j].eval()
    with torch.no_grad():
        for i in range(num_game_steps):
            # At each game step i, we run the model on 1000 examples
            src_test[:, i, :] = x_test[:,i,:]
            trg_test = y_test[:,i,:,:]
            for k in range(1000 // 50):
                preds = torch.zeros((50,1,25), device=device)
                preds_so_far = torch.zeros((50,5,25), device=device)
                for j in range(5):
                    preds = model[j](src_test[50*k:50*(k+1),:,:], preds_so_far, None, None)
                    loss_test = get_loss_wo_groundtruth(50, preds, trg_test[50*k:50*(k+1),:,:], device, j)
                    preds = F.softmax(preds, dim=-1)
                    _, ix = preds.data.topk(1)
                    for l in range(50):
                        preds_so_far[l, j, ix[l]] = 1
                    per_card_scores[i,j] += loss_test.item()
            for j in range(5):
                per_card_scores[i,j] /= (1000 // 50)
            
    return per_card_scores


def per_card_scores_online(x_test, y_test, model, batch):
    # x_test is 80 x 1000 x encod_dim
    # y_test is 80 x 1000 x 5 x decod_dim

    num_game_steps = torch.max(batch.seq_len).long().item()

    per_card_scores = np.zeros((num_game_steps,5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs = torch.zeros((80, 20, 858), device=device)
    trg = torch.zeros(20, 5, 26)

    preds = torch.zeros((1,1,26), device=device)
    preds_so_far = torch.zeros((1,5,26))
    preds_so_far_ = torch.zeros((5,1,5,26), device=device)

    # for j in range(5):
    model.eval()
    with torch.no_grad():
        for i in range(num_game_steps):
            count = 0
            for g in range(1000):
                if batch.seq_len[g] <= num_game_steps:
                    # At each game step i, we run the model on 1000 examples
                    obs[0:i+1,count%20,:] = x_test[0:i+1,g,:]
                    trg[count%20,:,:] = y_test[i,g,:,:]
                    count += 1

                    if count % 20 == 0:
                        obs *= 0
                        trg *= 0
                        preds *= 0
                        preds_so_far *= 0
                        preds_so_far_ *= 0
                        for j in range(5):
                            preds_so_far_[j,:] = preds_so_far.detach().clone()
                            # preds = model[j](obs.view(20, 80, 858), preds_so_far_[j,:], None, None)
                            preds = model(obs.view(20, 80, 858), preds_so_far_[j,:], None, None)
                            loss = get_loss_online(20, preds, trg, device, j)
                            per_card_scores[i,j] += loss.item()

            if count % 20 != 0:
                obs *= 0
                trg *= 0
                preds *= 0
                preds_so_far *= 0
                preds_so_far_ *= 0
                for j in range(5):
                    preds_so_far_[j,:] = preds_so_far.detach().clone()
                    # preds = model[j](obs[:,0:(count%20)+1,:].view(count, 80, 858), preds_so_far_[j,:], None, None)
                    preds = model(obs[:,0:(count%20)+1,:].view(count, 80, 858), preds_so_far_[j,:], None, None)
                    loss = get_loss_online(count, preds, trg[0:(count%20)+1,:,:], device, j)
                    per_card_scores[i,j] += loss.item()

            for j in range(5):
                per_card_scores[i,j] /= count
            
    return per_card_scores


