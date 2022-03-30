# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import matplotlib as mlp
mlp.use("Agg")

import time
import os
import sys
import argparse
import pprint

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch.nn.functional as F
import random
import math

from create import create_envs, create_threads, ActGroup
from eval import evaluate
import common_utils
import rela
import r2d2_lbs as r2d2
import utils

from transformer_embedding import get_model, Transformer
from train_funcs import get_loss_online
from per_card_scores import per_card_scores_online

def get_samples(batch, args):
    # obs = batch.obs["priv_s"].to("cpu")
    # obs = torch.cat([obs, batch.action["a"].to("cpu").unsqueeze(-2).repeat(1,1,args.num_player,1).float()], -1)
    # obs = torch.cat([obs, torch.zeros((80, batchsize, args.num_player, 18))], dim=3)
    # obs = obs[:,:,0,:]
    # # obs = obs.view(80, 2*batchsize, -1)

    batch_size = len(batch.seq_len)

    target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size,
                                        25)  # have to add empty card dimension to target!
    target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
    target = torch.cat([target, target_empty_mask.float()], -1)
    # for j in range(len(batch.seq_len)):
    #     target[batch.seq_len[j].long().item():, j, 0, :] *= 0
    #     target[batch.seq_len[j].long().item():, j, 1, :] *= 0
    # target = target.view(80, 2*batchsize, 5, -1)
    target = target[:,:,0,:,:]

    trgs = torch.zeros((batch_size, 7), device="cpu").type(torch.LongTensor)
    obs = torch.zeros(80, batch_size, 838).type(torch.LongTensor)

    obs = batch.obs["priv_s"][:,:,0,:]
    obs[:,:,433:783] = 0

    gamesteps = np.zeros(len(batch.seq_len))
    for j in range(len(batch.seq_len)):
        gamesteps[j] = random.randint(0, batch.seq_len[j]-1)
        obs[int(gamesteps[j])+1:,:] *= 0

    gamesteps = gamesteps.astype(int)

    # 0-27 is active agent's cards (25 is empty card, 26 is start of seq token, 27 is end of seq token)
    trgs[:,0] = 26
    trgs[:,6] = 27
    trgs[:,1:6] = (target[gamesteps, range(batch_size), :] == 1).nonzero(as_tuple=True)[2].reshape(-1,5)
    # for ex in range(batch_size):
    #     for card in range(1,6):
    #         if 1 in target[gamesteps[ex], ex, card-1, :]:
    #             trgs[ex,card] = int((target[gamesteps[ex], ex, card-1, :] == 1).nonzero(as_tuple=True)[0])
    #         else:
    #             trgs[ex,card] = 25

    return obs, trgs

def belief_run(model,obs,ck,target,total_losses,args,optim,step_num,stopwatch,device,train_or_eval):
    if train_or_eval == 'train':
        model.train()
        total_losses, losses = belief_run_loop(model,obs,ck,target,total_losses,args,optim,stopwatch,device,train_or_eval)
        loss = sum(losses)
        optim.zero_grad()
        loss.backward()
        stopwatch.time("forward & backward")
        if step_num < args.warm_up_period:
            for g in optim.param_groups:
                g['lr'] = step_num * args.lr / args.warm_up_period
        elif step_num == args.warm_up_period:
            for g in optim.param_groups:
                g['lr'] = args.lr
        else: # decay by inverse root schedule
            for g in optim.param_groups:
                # g['lr'] = args.lr * min(1.0/math.sqrt(step_num - args.warm_up_period), (step_num - args.warm_up_period) * (args.warm_up_period ** (-1.5)))
                g['lr'] = args.lr * 1.0/math.sqrt(step_num - args.warm_up_period)
        optim.step()
        step_num += 1
        stopwatch.time("update model")
        return total_losses, step_num
    else:
        with torch.no_grad():
            model.eval()
            total_losses, _ = belief_run_loop(model,obs,ck,target,total_losses,args,optim,stopwatch,device,train_or_eval)
            return total_losses
    
def belief_run_loop(model,obs,ck,target,total_losses,args,optim,stopwatch,device,train_or_eval):
    # gamesteps = np.zeros(len(batch.seq_len))

    # for j in range(len(batch.seq_len)):
    #     seq_len = batch.seq_len[j]

    #     gamestep = random.randint(0, seq_len-1)
    #     gamesteps[j] = gamestep
    #     obs[gamestep+1:, j, :] *= 0

        # gamestep = random.randint(0, seq_len-1)
        # gamesteps[j+args.batchsize] = gamestep
        # obs[gamestep+1:, j+args.batchsize, :] *= 0

    # gamesteps = gamesteps.astype(int)

    # rand_perm = torch.randperm(2*args.batchsize)
    # obs       = obs[:,rand_perm,:]
    # target    = target[:,rand_perm,:]
    # gamesteps = gamesteps[rand_perm]

    _, trg_mask = create_masks(obs, target[:, :-1])

    losses = []
    target = target.to(device)
    obs = torch.transpose(obs, 0, 1)
    obs = obs.to(device)
    # ck = ck.to(device)
    trg_mask = trg_mask.to(device)

    preds = model(obs, None, target[:, :-1], None, trg_mask)#.to("cpu")

    for j in range(5):
        loss = F.cross_entropy(preds[:,j,:].view(-1, preds.size(-1)), target[:,j+1].contiguous().view(-1), ignore_index = 5606)
        total_losses[j] += loss.item()
        losses.append(loss)

    # for j in range(5):
    #     trg = target
    #     trg[:, j:] *= 0
    #     print(torch.cuda.memory_allocated())
    #     preds = model(obs, trg, None, None)[:,j,:]#.to("cpu")
    #     loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target[:,j].contiguous().view(-1), ignore_index = 5606)
    #     losses.append(loss)
    #     total_losses[j] += loss.item()

        # trg = torch.zeros(args.batchsize, 5, 26)
        # for k in range(args.batchsize):
        #     trg[k,:,:] = target[gamesteps[k],k,:,:]

    # loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target.contiguous().view(-1), ignore_index = -1)

    # loss = get_loss_online(args.batchsize, preds, trg, device, j)

        # losses.append(loss)
        
        # total_losses[j] += loss.item()

        # preds_so_far[:,j,:] = F.softmax(preds.view(args.batchsize,26), dim=-1)

    return total_losses, losses


def create_masks(input_seq, target_seq):
    # creates mask with 0s wherever there is padding in the input
    input_pad = 5606
    input_msk = (input_seq != input_pad).unsqueeze(1)

    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)

    return input_msk, nopeak_mask

# def free_memory(model, obs, preds, preds_so_far_):
#     torch.cuda.empty_cache()
#     for j in range(5):
#         model[j] = model[j].to(device)
#     obs = obs.to(device)
#     preds = preds.to(device)
#     preds_so_far_ = preds_so_far_.to(device)

def rainman(last_actions, other_player_hands):
    # last_actions is 2d array, seq_length x 55, where 55 is encoding dim for 2 player last action
    # other_player_hands is 2d array, seq_length x 125, where 125 is the 5 x 25, handsize and card embedding dim respectively
    # card to predict is which card in the hand we are tracking the belief of (0 to 4)
    
    seq_length = len(last_actions)
    
    cards_left = np.array([3,2,2,2,1,3,2,2,2,1,3,2,2,2,1,3,2,2,2,1,3,2,2,2,1]) # colour-major-ordering
    
    # own_hand_tracking is handsize x 2 x 5, where 2 is [colour, rank] and 5 is the colour/rank dim
    # 1 at index denotes the given colour/rank for the index is possible
    own_hand_tracking = np.ones((5, 2, 5))
    
    # flag on cards we exactly know, so as to avoid double subtracting over iterations
    exactly_know = np.zeros(5)
    
    # account for other player's initial hand
    other_player_init_hand = other_player_hands[0]
    for j in range(5):
        card = list(other_player_init_hand[25 * j : 25 * (j+1)]).index(1)
        cards_left[card] -= 1
    
    for i in range(1, seq_length):
        last_action = last_actions[i]
        
        if last_action[0] == 1: # active player's turn
            if last_action[2] == 1 or last_action[3] == 1: # active player plays or discards
                which_card_in_hand = list(last_action[23:28]).index(1)
                if exactly_know[which_card_in_hand] == 0: 
                    # card is not yet accounted for in cards_left, so we thus account for it
                    which_card = list(last_action[28:53]).index(1)
                    cards_left[which_card] -= 1
                # reset what we know about the card slot
                own_hand_tracking[which_card_in_hand, :, :] = np.ones((2,5))
                exactly_know[which_card_in_hand] = 0
                # shift cards
                own_hand_tracking[which_card_in_hand:, :, :] = np.roll(own_hand_tracking[which_card_in_hand:, :, :], -1, 0)
                exactly_know[which_card_in_hand:] = np.roll(exactly_know[which_card_in_hand:], -1, 0)
                
        else: # other player's turn
            if last_action[2] == 1 or last_action[3] == 1: # other player plays or discards      
                # account for other player's new card
                try:
                    new_card = list(other_player_hands[i, 100:125]).index(1)
                    cards_left[new_card] -= 1
                except:
                    try:
                        new_card = list(other_player_hands[i, 75:100]).index(1)
                        cards_left[new_card] -= 1
                    except:
                        pass
                            
            elif last_action[4] == 1: # other player gives colour hint
                which_colour = list(last_action[8:13]).index(1)
                for k in range(5):
                    if last_action[k + 18] == 1: # set all colours besides which_colour to 0 for kth card
                        indices = list([0,1,2,3,4])
                        del indices[which_colour]
                        own_hand_tracking[k, 0, indices] = 0
                    else: # set which_colour to 0 for kth card
                        own_hand_tracking[k, 0, which_colour] = 0
                    if exactly_know[k] == 0 and list(own_hand_tracking[k, 0, :]).count(1) == 1 and list(own_hand_tracking[k, 1, :]).count(1) == 1:
                        # exactly know kth card and haven't accounted for it yet, so we thus account for it
                        exactly_know[k] = 1
                        cards_left[list(own_hand_tracking[k, 0, :]).index(1) * 5 
                                   + list(own_hand_tracking[k, 1, :]).index(1)] -= 1
                        
            else: # other player gives rank hint
                which_rank = list(last_action[13:18]).index(1)
                for k in range(5):
                    if last_action[k + 18] == 1: # set all ranks besides which_rank to 0 for kth card
                        indices = list([0,1,2,3,4])
                        del indices[which_rank]
                        own_hand_tracking[k, 1, indices] = 0
                    else: # set which_rank to 0 for kth card
                        own_hand_tracking[k, 1, which_rank] = 0
                    if exactly_know[k] == 0 and list(own_hand_tracking[k, 0, :]).count(1) == 1 and list(own_hand_tracking[k, 1, :]).count(1) == 1:
                        # exactly know kth card and haven't accounted for it yet, so we now account for it
                        exactly_know[k] = 1
                        cards_left[list(own_hand_tracking[k, 0, :]).index(1) * 5 
                                   + list(own_hand_tracking[k, 1, :]).index(1)] -= 1 
                        
    # having processed history, narrow down the cards:

    rainman_prob = np.zeros((25,5))
    for j in range(5):
        rainman_prob[:,j] = cards_left

    for card_to_predict in range(5):
        for j in range(5): # narrow down the colours
            if own_hand_tracking[card_to_predict, 0, j] == 0:
                for k in range(5): # set all ranks k of colour j to 0 in cards_left
                    rainman_prob[j * 5 + k, card_to_predict] = 0
    
    for card_to_predict in range(5):
        for j in range(5): # narrow down the ranks
            if own_hand_tracking[card_to_predict, 1, j] == 0:
                for k in range(5): # set all colours k of rank j to 0 in cards_left
                    rainman_prob[k * 5 + j, card_to_predict] = 0
                
    return rainman_prob

# enumeration of 1 colour (size 216 (=4*3*3*3*2) dictionary)
def enumerate_v0():
    enumeration = {}
    count = 0
    for a in range(4):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(2):
                        enumeration[(a,b,c,d,e)] = count
                        count += 1
    return enumeration

# enumeration of hints that got coloured
def enumerate_hints():
    enumeration = {}
    count = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for e in range(2):
                        enumeration[(a,b,c,d,e)] = count
                        count += 1
    return enumeration
        

def load_op_model(method, idx1, idx2, device):
    """load op models, op models was trained only for 2 player
    """
    #root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # assume model saved in root/models/op
    #folder = os.path.join(root, "models", "op", method)
    folder = os.path.join("models", "op", "sad")
    agents = []
    for idx in [idx1, idx2]:
        if idx >= 0 and idx < 3:
            num_fc = 1
            skip_connect = False
        elif idx >= 3 and idx < 6:
            num_fc = 1
            skip_connect = True
        elif idx >= 6 and idx < 9:
            num_fc = 2
            skip_connect = False
        else:
            num_fc = 2
            skip_connect = True
        weight_file = os.path.join(folder, f"M{idx}.pthw")
        if not os.path.exists(weight_file):
            print(f"Cannot find weight at: {weight_file}")
            assert False

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]
        agent = r2d2.R2D2Agent(
            True, # False easier to use as VDN! if using False set method=iql!
            3,
            0.999,
            0.9,
            device,
            input_dim,
            hid_dim,
            output_dim,
            2,
            5,
            False,
            num_fc_layer=num_fc,
            skip_connect=skip_connect,
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents



def parse_args():
    parser = argparse.ArgumentParser(description="train belief model")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_obs", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--pred_weight", type=float, default=0)
    parser.add_argument("--num_eps", type=int, default=80)

    parser.add_argument("--load_model", type=str, default="")

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--eta", type=float, default=0.9, help="eta for aggregate priority")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=1)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)
    parser.add_argument("--encoder_dim", type=int, default=858)
    parser.add_argument("--decoder_dim", type=int, default=26)
    parser.add_argument("--num_heads", type=int, default=8) #must have num_head | encoder_dim, decoder_dim
    parser.add_argument("--N", type=int, default=6)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--warm_up_period", type=float, default=100000, help="Warm Up Period")
    parser.add_argument("--eps", type=float, default=1e-9, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda")
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--eval_epochs", type=int, default=1)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settingsœ
    parser.add_argument("--burn_in_frames", type=int, default=2500) #2500
    parser.add_argument("--replay_buffer_size", type=int, default=5000) #5000
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    #MOD 3: flag that switches off replay buffer priority by default
    parser.add_argument(
        "--no_replay_buffer_priority", type=bool, default=True, help="switch replay buffer priority off"
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=15, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    # special modes
    parser.add_argument("--obs_n_most_recent_own_cards", type=int, default=0)
    parser.add_argument("--use_softmax_policy", type=int, default=0)
    parser.add_argument("--log_beta_range", type=str, default="0.5,10")
    parser.add_argument("--eval_log_betas", type=str, default="1,2,3,5,7,10")
    parser.add_argument("--q_variant", type=str, default="doubleq")
    # Feature 26: empty card
    parser.add_argument("--card-feature-dim", type=int, default=26, help="dimensionality of a single card feature")
    parser.add_argument("--use_autoregressive_belief", type=int, help="if True use autoregressive belief")
    parser.add_argument("--belief_type", type=str, default="own_hand", help="if True use autoregressive belief") # can be own_hand or public
    parser.add_argument("--model_is_op", type=bool, default=False,
                        help="whether OP model is used")  # can be own_hand or public
    parser.add_argument("--idx", default=1, type=int, help="which model to use (for OP)?")

    args = parser.parse_args()
    assert args.method in ["vdn", "iql"]
    # assert args.load_model != "", "You need to load a model in train LBS mode!"
    return args


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    args.load_model = "models/aux/aux_2p_13.pthw"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    # pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_eps
    )
    expected_eps = np.mean(explore_eps)
    # print("explore eps:", explore_eps)
    # print("avg explore eps:", np.mean(explore_eps))

    # if not not args.model_is_op:

    # COMMENTED FROM HERE FOR SINGLE REPLAY v v v

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.hand_size,
        args.train_bomb,
        explore_eps,
        args.max_len,
        args.sad,
        args.shuffle_obs,
        args.shuffle_color,
        #args.obs_n_most_recent_own_cards # modification by CHRISTIAN
    )

    # full obs modification (CHRISTIAN)
    feature_size = games[0].feature_size()
    if args.belief_type == "public":
        # remove all card observations from input
        feature_size = feature_size - (args.hand_size*args.num_player*args.card_feature_dim + 5)
    print("FEATURE SIZE: ", feature_size)

    # MOD 5: replace weight loading with agent loading!
    if args.load_model != "" and not args.model_is_op:
        print("*****loading pretrained model*****")
        # utils.load_weight(agent.online_net, args.load_model, args.train_device)
        overwrite = {}
        overwrite["vdn"] = (args.method == "vdn")
        overwrite["device"] = "cuda:0"
        overwrite["boltzmann_act"] = False
        agent, cfg = utils.load_agent(
            args.load_model,
            overwrite,
        )
        agent.log_trajectories = True
        print("CFG: ", cfg)
        assert cfg["num_player"] == args.num_player, "Model num players does not coincide with config num players!"
        print("*****done*****")
    # elif args.load_model != "" and args.model_is_op:
    # agent = load_op_model(args.method, args.idx, args.idx, args.train_device)[0]

    # else:
    # agent = r2d2.R2D2Agent(
    #     (args.method == "vdn"),
    #     args.multi_step,
    #     args.gamma,
    #     args.eta,
    #     args.train_device,
    #     feature_size, # if not args.use_softmax_policy else feature_size + 1,
    #     args.rnn_hid_dim,
    #     games[0].num_action(),
    #     args.num_lstm_layer,
    #     args.hand_size,
    #     args.no_replay_buffer_priority  # uniform priority
    #     # args.use_softmax_policy,
    #     # args.q_variant
    # )

    agent.sync_target_with_online()

    if args.load_model and not args.model_is_op:
        print("*****loading pretrained model*****")
        # utils.load_weight(agent.online_net, args.load_model, args.train_device)
        utils.load_weight(agent.online_net, args.load_model, "cpu")
        print("*****done*****")

    agent = agent.to(args.train_device)
    # agent = agent.to("cpu")
    # optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    eval_agent = agent.clone(args.train_device, {"vdn": False})
    # eval_agent = agent.clone("cpu", {"vdn": False})

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent if not args.no_replay_buffer_priority else 0.0,
        args.priority_weight if not args.no_replay_buffer_priority else 1.0,
        args.prefetch,
    )

    act_group = ActGroup(
        args.method,
        args.act_device,
        agent,
        args.num_thread,
        args.num_game_per_thread,
        args.multi_step,
        args.gamma,
        args.eta,
        args.max_len,
        args.num_player,
        replay_buffer,
    )

    assert args.shuffle_obs == False, 'not working with 2nd order aux'
    context, threads = create_threads(
        args.num_thread, args.num_game_per_thread, act_group.actors, games,
        #use_softmax_policy=args.use_softmax_policy,
        #betas_range=torch.Tensor([float(x) for x in args.log_beta_range.split(",")])
    )
    act_group.start()
    # else: # OP MODEL!
    #     agents = load_op_model(args.method, args.idx, args.idx, args.train_device)
    #     if agents is not None:
    #         runners = [rela.BatchRunner(agent, args.train_device, 1000, ["act"]) for agent in agents]
    #     num_player = len(runners)
    #
    #     context = rela.Context()
    #     games = create_envs(
    #         args.num_thread * args.num_game_per_thread,
    #         args.seed,
    #         args.num_player,
    #         args.hand_size,
    #         args.train_bomb,
    #         explore_eps,
    #         -1,
    #         True, # sad flag
    #         False,
    #         False,
    #     )
    #
    #     for g in games:
    #         env = hanalearn.HanabiVecEnv()
    #         env.append(g)
    #         actors = []
    #         for i in range(num_player):
    #             actors.append(rela.R2D2Actor(runners[i], 1))
    #         thread = hanalearn.HanabiThreadLoop(actors, env, True)
    #         context.push_env_thread(thread)
    #
    #     for runner in runners:
    #         runner.start()

    context.start()
    
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)


    # COMMENTED TILL HERE FOR SINGLE REPLAY ^ ^ ^

    # time.sleep(5)
    # while True:
    #     batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
    #     print(batch.seq_len)
    #     priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
    #     priority = rela.aggregate_priority(
    #         priority.cpu(), batch.seq_len.cpu(), args.eta
    #     )
    #     replay_buffer.update_priority(priority)
    #     time.sleep(2)

    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    # stat = common_utils.MultiCounter(args.save_dir)
    # tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()

    # count_duplicates = 0
    # samples_seen = []
    # for j in range(100000//32):
    #     print(str(j+1)+"/"+str(100000//32))
    #     print(count_duplicates)
    #     batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
    #     obs = batch.obs["priv_s"].to("cpu")
    #     obs = torch.cat([obs, batch.action["a"].to("cpu").unsqueeze(-2).repeat(1,1,args.num_player,1).float()], -1)
    #     obs = torch.cat([obs, torch.zeros((80, args.batchsize, args.num_player, 18))], dim=3)
    #     obs = obs[:,:,0,:]
    #     for k in range(32):
    #         if list(obs[0,k,:]) not in samples_seen:
    #             samples_seen.append(obs[0,k,:])
    #         else:
    #             count_duplicates += 1

    #     priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
    #     priority = rela.aggregate_priority(
    #         priority.cpu(), batch.seq_len.cpu(), args.eta
    #     )
    #     replay_buffer.update_priority(priority)
    #     torch.cuda.synchronize()
    # print(colours)

    ##############
    # Create LBS model
    #############

    # model = [get_model(args.encoder_dim, args.decoder_dim, args.N, args.num_heads).to(device) for _ in range(5)]
    model = get_model(206, 28, 256, args.N, args.num_heads).to(device)
    # model = Transformer(206, 28, 256, args.N, args.num_heads)
    # model.load_state_dict(torch.load("saves_while_training/model$_2player_belief_trainedsad6.pth"))
    # model = model.to(device)
    # model = Transformer(5607, 28, 256, args.N, args.num_heads)
    # model.load_state_dict(torch.load("saves_while_training/model$_2player_belief_online_w_v0_pr.pth"))
    # model = model.to(device)

    # # optim = [torch.optim.Adam(model[j].parameters(), lr=args.lr, betas=(0.9, 0.98), eps=args.eps) for j in range(5)]
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.998), eps=args.eps)

    ##############
    # END Create LBS model
    #############
    # print("Doing per card scores...")

    # batch, weight = replay_buffer.sample(1000, args.train_device)
    # x_test, y_test = get_samples(batch, 1000)
    # asdf = per_card_scores_online(x_test, y_test, model, batch)

    # print("Done per card scores!")

    # v0_enumeration = enumerate_v0()
    # hint_enumeration = enumerate_hints()

    # preds = torch.zeros((args.batchsize,1,26), device=device)
    # preds_so_far = torch.zeros((args.batchsize,5,26), device=device)

    # preds_test = torch.zeros((args.batchsize,1,26), device=device)
    # preds_so_far_test = torch.zeros((args.batchsize,5,26))
    # preds_so_far_test_ = torch.zeros((5,args.batchsize,5,26), device=device)

    # obs = torch.zeros((80, args.batchsize, 838)).type(torch.LongTensor)
    # # ck = torch.zeros((args.batchsize, 2000)).type(torch.LongTensor)
    # target = torch.zeros((args.batchsize, args.hand_size+2)).type(torch.LongTensor)

    # obses = torch.zeros((args.batchsize*2, 1200)).type(torch.LongTensor)
    # # cks = torch.zeros((args.batchsize*4, 2000)).type(torch.LongTensor)
    # targets = torch.zeros((args.batchsize*2, args.hand_size+2)).type(torch.LongTensor)

    total_losses = np.zeros(5)
    # total_tests  = np.zeros(5)

    print_every = 50
    save_loss_every = 250
    save_entr_vs_timestep_every = 3000
    running_losses = [[] for _ in range(5)]
    # running_tests  = [[] for _ in range(5)]

    # best_model_scores = [100 for _ in range(5)]
    best_model_score = 100

    step_num = 1

    start = time.time()
    temp = start

    for epoch in range(args.num_epoch):
        print("beginning of epoch: " + str(epoch+1))
        print(common_utils.get_mem_usage())
        # tachometer.start()
        # stat.reset()
        stopwatch.reset()
        loss_lst_train = None

        for batch_idx in range(args.epoch_len):
            if epoch == -1: # DBG
                break
            # num_update = batch_idx + epoch * args.epoch_len
            # if num_update % args.num_update_between_sync == 0:
            #     agent.sync_target_with_online()
            # if num_update % args.actor_sync_freq == 0:
            #     act_group.update_model(agent)

            # if batch_idx % 50 == 0:
            #     time.sleep(1)

            # torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)

            if args.belief_type == "public":
                obs = batch.obs["priv_s"][:, :, -feature_size:] if args.num_player < 5 else batch.obs["priv_s"][:, :, 0, -feature_size:]
                # if args.num_player < 5:
                #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-1).float()], -1)
                # else:
                obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1, 1, args.num_player, 1).float()], -1)
                obs = obs[:torch.max(batch.seq_len).long().item()]
                target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size, 25) # have to add empty card dimension to target!
                target_empty_mask = (target.sum(-1, keepdim=True)==0.0)
                target = torch.cat([target, target_empty_mask.float()], -1)
                target = target[:torch.max(batch.seq_len).long().item()]
            elif args.belief_type == "own_hand":
                
                batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
                obs, target = get_samples(batch, args)
                priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
                priority = rela.aggregate_priority(
                    priority.cpu(), batch.seq_len.cpu(), args.eta
                )
                replay_buffer.update_priority(priority)

                stopwatch.time("sample data")

            else:
                assert False, "Unknown belief type: {}".format(args.belief_type)
            # loss_lst_train, priority = agent.loss_lbs(lbs_net, obs, target, stat) # , args.pred_weight,
            # loss_lst_train_lst.append( loss_lst_train.detach().cpu())

            # MOD2: ReplayBuffer aggregate_priority not needed
            # preds *= 0
            # preds_so_far *= 0

            total_losses, step_num = belief_run(model,obs,None,target,total_losses,args,optim,step_num,stopwatch,device,'train')

            # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
            # priority = rela.aggregate_priority(
            #     priority.cpu(), batch.seq_len.cpu(), args.eta
            # )
            # replay_buffer.update_priority(priority)
            # stopwatch.time("updating priority")
            torch.cuda.synchronize()

            # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            # obs[:], target[:] = get_samples(batch, args.batchsize)[:]
            
            # preds *= 0
            # preds_so_far *= 0
            # preds_so_far_ *= 0

            # _, total_tests = belief_run(model,obs,target,batch,total_tests,args,preds,preds_so_far,preds_so_far_,'eval')

            # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
            # priority = rela.aggregate_priority(
            #     priority.cpu(), batch.seq_len.cpu(), args.eta
            # )
            # replay_buffer.update_priority(priority)
            # torch.cuda.synchronize()

            # loss = (loss_lst_train).mean()
            # loss.backward()

            # load another policy in here
            

            # g_norm = torch.nn.utils.clip_grad_norm_(
            #     agent.online_net.parameters(), args.grad_clip
            # )
            # optim.step()
            # optim.zero_grad()

            # MOD1: ReplayBuffer update_priotity is not needed

            # stat["loss"].feed(loss.detach().item())
            # stat["grad_norm"].feed(g_norm)

            if (batch_idx + 1) % print_every == 0:
                loss_avgs = [total_losses[j] / print_every for j in range(5)]
                # loss_test_avgs = [total_tests[j] / print_every for j in range(5)]
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, batch_idx + 1, 0.2*sum(loss_avgs), time.time() - temp,
                print_every))
                # print("1st card loss = %.3f, 1st card test = %.3f, 2nd card loss = %.3f, 2nd card test = %.3f, 3rd card loss = %.3f, 3rd card test = %.3f, 4th card loss = %.3f, 4th card test = %.3f, 5th card loss = %.3f, 5th card test = %.3f" % 
                # (loss_avgs[0], loss_test_avgs[0], loss_avgs[1], loss_test_avgs[1], loss_avgs[2], loss_test_avgs[2], loss_avgs[3], loss_test_avgs[3], loss_avgs[4], loss_test_avgs[4]))

                for j in range(5):
                    running_losses[j].append(loss_avgs[j])
                    # running_tests[j].append(loss_test_avgs[j])
                # for j in range(5):
                if 0.2*sum(loss_avgs) < best_model_score:
                    # torch.save(model[j].state_dict(),"/pyhanabi/saves_while_training/model"+str(j)+"_2player_belief_online.pth")
                    torch.save(model.state_dict(),"/pyhanabi/saves_while_training/model$_2player_belief_fully_conn.pth")
                    best_model_score = 0.2*sum(loss_avgs)

                total_losses = np.zeros(5)
                total_tests  = np.zeros(5)
                temp = time.time()

            if (batch_idx + 1) % (save_loss_every) == 0:
                np.save(os.path.join('saves_while_training', 'train_loss_fully_conn.npy'), np.array(running_losses))
                # for j in range(5):
                    # np.save(os.path.join('saves_while_training', 'train_loss'+str(j)+'_online.npy'), np.array(running_losses[j]))
                    # np.save(os.path.join('saves_while_training', 'valid_loss'+str(j)+'_online.npy'), np.array(running_tests[j]))
            # if (batch_idx + 1) % (save_entr_vs_timestep_every) == 0:
            #     print("Calculating per card cross entropies...")
            #     batch, weight = replay_buffer.sample(1000, args.train_device)
            #     x_test, y_test = get_samples(batch, args)
            #     # np.save(os.path.join('saves_while_training', 'per_card_scores_online.npy'), per_card_scores_online(x_test, y_test, model, batch))
            #     print("Calculated.")

        stopwatch.summary()

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: " + str(epoch+1))
        # tachometer.lap(
        #     act_group.actors, replay_buffer, args.epoch_len * args.batchsize, count_factor
        # )
        # stopwatch.summary()
        # stat.summary(epoch)

        # context.pause()

        # eval_seed = (9917 + epoch * 999999) % 7777777
        # eval_agent.load_state_dict(agent.state_dict())

        # print("EVALUATE - START")
        # print([eval_agent for _ in range(args.num_player)],
        #     1000,
        #     eval_seed,
        #     args.eval_bomb,
        #     0,  # explore eps
        #     args.sad,
        #     args.obs_n_most_recent_own_cards,
        #     args.hand_size,
        #     args.use_softmax_policy,
        #     [float(b) for b in args.eval_log_betas.split(",")])
        # print("EVALUATE - END")
        # quit()

        # print("Evaluating policy performance...")
        # d = evaluate(
        #     [eval_agent for _ in range(args.num_player)],
        #     1000,
        #     eval_seed,
        #     args.eval_bomb,
        #     0,  # explore eps
        #     args.sad,
        #     #args.obs_n_most_recent_own_cards,
        #     hand_size=args.hand_size,
        # )
        torch.cuda.synchronize()

        # commented from here epoch -1 break code v v v

        # print("Evaluating belief model performance...")
        # stopwatch.time("sync and updating")

        # # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)

        # if args.belief_type == "public":#
        #     assert False, "public belief not currently supported!"
        #     # if args.num_player < 5:
        #     #     obs = batch.obs["priv_s"][:, :, -feature_size:]
        #     #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-1).float()], -1)
        #     #     target = batch.obs["own_hand"].view(batch.obs["own_hand"].shape[0],
        #     #                                         batch.obs["own_hand"].shape[1],
        #     #                                         args.hand_size * args.num_player * args.card_feature_dim)
        #     # else:
        #     obs = batch.obs["priv_s"][:, :, 0, -feature_size:]
        #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1, 1, args.num_player, 1).float()], -1)
        #     # TODO: This does not work for public belief!
        #     target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size,
        #                                         25)  # have to add empty card dimension to target!
        #     target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
        #     target = torch.cat([target, target_empty_mask.float()], -1)
        #     obs = obs[:torch.max(batch.seq_len).long().item()]
        #     target = target[:torch.max(batch.seq_len).long().item()]
        # elif args.belief_type == "own_hand":

        #     obs *= 0
        #     target *= 0
        #     ck *= 0
        #     obs[:] = obses[(batch_idx % 25)*args.batchsize:(1 + batch_idx % 25)*args.batchsize,:]
        #     ck[:]  = cks[(batch_idx % 25)*args.batchsize:(1 + batch_idx % 25)*args.batchsize,:]
        #     target[:] = targets[(batch_idx % 25)*args.batchsize:(1 + batch_idx % 25)*args.batchsize,:]
        #     # obs[:], ck[:], target[:] = get_samples(batch, args)[:]
        #     stopwatch.time("sample data")

        # else:
        #     assert False, "Unknown belief type: {}".format(args.belief_type)
        # # loss_lst, priority = agent.loss_lbs(lbs_net, obs, target, stat, eval=True)  # , args.pred_weight,
        # # loss_lst_lst.append(loss_lst.detach().cpu())

        # # preds *= 0
        # # preds_so_far *= 0

        # total_losses = belief_run(model,obs,ck,target,total_losses,args,optim,stopwatch,device,'train')

        # # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
        # # priority = rela.aggregate_priority(
        # #     priority.cpu(), batch.seq_len.cpu(), args.eta
        # # )
        # # replay_buffer.update_priority(priority)
        # torch.cuda.synchronize()

        # # batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
        # # obs[:], target[:] = get_samples(batch, args.batchsize)[:]

        # # preds_test *= 0
        # # preds_so_far_test *= 0
        # # preds_so_far_test_ *= 0
        
        # # _, total_tests = belief_run(model,obs,target,batch,total_tests,args,preds_test,preds_so_far_test,preds_so_far_test_,'eval')

        # # priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
        # # priority = rela.aggregate_priority(
        # #     priority.cpu(), batch.seq_len.cpu(), args.eta
        # # )
        # # replay_buffer.update_priority(priority)
        # # torch.cuda.synchronize()

        # # for j in range(5):
        # #     optim[j].zero_grad()
        # # loss = sum(losses)
        # # loss.backward()
        # # for j in range(5):
        # #     optim[j].step()


        # # MOD2: ReplayBuffer aggregate_priority not needed

        # # if loss_lst_train is not None:
        # #     loss_train = loss_lst_train.mean().detach()
        # #     print("CrossEntropyLoss: ", loss_train)

        # # loss = loss_lst.mean().detach()
        # # print("Category Loss: ", loss)

        # loss_avgs = [total_losses[j] for j in range(5)]
        # # loss_test_avgs = [total_tests[j]for j in range(5)]
        # print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
        # epoch + 1, batch_idx + 1, 0.2*sum(loss_avgs), time.time() - temp,
        # 1))
        # # print("1st card loss = %.3f, 1st card test = %.3f, 2nd card loss = %.3f, 2nd card test = %.3f, 3rd card loss = %.3f, 3rd card test = %.3f, 4th card loss = %.3f, 4th card test = %.3f, 5th card loss = %.3f, 5th card test = %.3f" % 
        # # (loss_avgs[0], loss_test_avgs[0], loss_avgs[1], loss_test_avgs[1], loss_avgs[2], loss_test_avgs[2], loss_avgs[3], loss_test_avgs[3], loss_avgs[4], loss_test_avgs[4]))
        # total_losses = np.zeros(5)
        # # total_tests  = np.zeros(5)
        # temp = time.time()

        # commented till here epoch -1 break code ^ ^ ^

        # if epoch > 0 and epoch % 50 == 0:
        #     force_save_name = "model_epoch%d" % epoch
        # else:
        #     force_save_name = None

        # if epoch == -1 or epoch%args.eval_epochs == 0:
        #     print("Saving belief model...")
        #     torch.save({"LBSBeliefNet": lbs_net.state_dict()}, os.path.join(args.save_dir, "lbs_model_ep{}.pthw".format(epoch)))
        #     print("Saving cross entropy graph...")
        #     import matplotlib.pyplot as plt

        #     loss_lst = loss_lst.detach().view(loss_lst.shape[0], -1).mean(-1).cpu()
        #     print("crossentropy data (val - hard):")
        #     print(loss_lst)

            # Plot average cross-entropy per step
            # plt.plot(list(range(len(loss_lst))), loss_lst, label="val")
            # if loss_lst_train is not None:
            #     loss_lst_train = loss_lst_train.detach().view(loss_lst_train.shape[0], -1).mean(-1).cpu()
            #     plt.plot(list(range(len(loss_lst_train))), loss_lst_train, label="train")
            #     print("crossentropy data (train - soft):")
            #     print(loss_lst_train)

            # plt.xlabel('steps')
            # plt.ylabel('cross entropy (whole hand)')
            # plt.title("Epoch {}".format(epoch))
            # plt.legend()
            # plt.savefig(os.path.join(args.save_dir, "lbs_model_ep{}.png".format(epoch)), bbox_inches='tight')
            # plt.clf()

            # Train loss over time plot
            # if loss_lst_train_lst:
            #     #lt = torch.stack(loss_lst_train_lst, 0)
            #     # lt = pad_sequence(loss_lst_train_lst, batch_first=True).permute(1,0,2,3)
            #     # lt = lt.view(lt.shape[0], -1).mean(-1, keepdim=True)
            #     _lt = [ _l.mean() for _l in loss_lst_train_lst]
            #     plt.plot(list(range(len(loss_lst_train_lst))), _lt, label="Train cross-entropy loss")
            #     # for i, _lt in enumerate(lt):
            #     #    plt.plot(list(range(i+1)), _lt, label="game steps {}".format(i))
            #     plt.xlabel('training episodes')
            #     plt.ylabel('cross entropy (whole hand)')
            #     plt.title("Epoch {}".format(epoch))
            #     plt.legend()
            #     plt.savefig(os.path.join(args.save_dir, "lbs_model_trainloss_ep{}.png".format(epoch)), bbox_inches='tight')
            #     plt.clf()

            # Eval loss over time plot
            # if loss_lst_lst:
            #     #lt = pad_sequence(loss_lst_lst, batch_first=True)
            #     # lt = pad_sequence(loss_lst_lst, batch_first=True).permute(1, 0, 2, 3)
            #     # lt = lt.view(lt.shape[0], -1).mean(-1, keepdim=True)
            #     _lt = [ _l.mean() for _l in loss_lst_lst]
            #     plt.plot(list(range(len(loss_lst_lst))), _lt, label="Eval category loss")
            #     plt.xlabel('training episodes')
            #     plt.ylabel('cross entropy (whole hand)')
            #     plt.title("Epoch {}".format(epoch))
            #     plt.legend()
            #     plt.savefig(os.path.join(args.save_dir, "lbs_model_valloss_ep{}.png".format(epoch)), bbox_inches='tight')
            #     plt.clf()

            #score, score_std, perfect, *_ = d
            # scores_mean = d["scores_mean"]
            # scores_std = d["scores_std"]
            # fraction_perfect = d["fraction_perfect"]
            # #model_saved = saver.save(
            # #    None, agent.online_net.state_dict(), scores_mean, force_save_name=force_save_name
            # #)
            # print(
            #     "epoch %d, eval score: %.4f, eval score std: %.4f, fraction perfect: %.2f"
            #     % (epoch, scores_mean, scores_std, fraction_perfect * 100)
            # )

        # context.resume()
        print("==========")
