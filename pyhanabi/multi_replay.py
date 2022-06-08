# train_lbs_embedding
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
import string

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch.nn.functional as F
import random

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

    trinary = False

    batch_size = len(batch.seq_len)


    if trinary:
        target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size,
                                            3)  # have to add empty card dimension to target!
    else:
        target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:2], args.hand_size,
                                            25)  # have to add empty card dimension to target!
    target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
    target = torch.cat([target, target_empty_mask.float()], -1)

    target = target.view(80, batch_size, 5, -1)
    # target = target[:,:,1,:,:]

    srcs = torch.zeros((batch_size, 3200), dtype=torch.long)
    trgs = torch.zeros((batch_size, 7))

    gamesteps = np.zeros(batch_size)
    for j in range(len(batch.seq_len)):
        gamesteps[j] = random.randint(0, batch.seq_len[j]-1)
        #gamesteps[j + len(batch.seq_len)] = random.randint(0, batch.seq_len[j]-1)

    gamesteps = gamesteps.astype(int)

    if trinary:
        # 0-5 is active agent's cards (3 is empty card, 4 is start of seq token, 5 is end of seq token)
        trgs[:,0] = 4
        trgs[:,6] = 5
        trgs[:,1:6] = (target[gamesteps, range(batch_size), :] == 1).nonzero(as_tuple=True)[2].reshape(-1,5)
    else:
        # 0-27 is active agent's cards (25 is empty card, 26 is start of seq token, 27 is end of seq token)
        trgs[:,0] = 26
        trgs[:,6] = 27
        trgs[:,1:6] = (target[gamesteps, range(batch_size), :] == 1).nonzero(as_tuple=True)[2].reshape(-1,5)

    obs = batch.obs["priv_s"]
    # obs = obs[:, :, 1, :]
    obs = obs.view(80, batch_size, -1)
    start = time.time()

    # 0-25 is partner's cards
            #   0-25 is first card
            #   26-51 is second card
            #   52-77 is third card
            #   78-103 is fourth card
            #   104-129 is fifth card
    
    partner_cards = obs[:,:,125:250].reshape(80,batch_size,5,25)
    partner_cards_empty_mask = (partner_cards.sum(-1, keepdim=True) == 0.0)
    partner_cards = torch.cat([partner_cards, partner_cards_empty_mask.float()], -1)
    partner_cards = (partner_cards == 1).nonzero(as_tuple=True)[3].reshape(80,batch_size,5)

    # 26-66 is remaining deck size

    decksizes = 26 + torch.sum(obs[:,:,252:292], -1, dtype = torch.long)

    # 67+5*c-72+5*c is fireworks of colour c

    fireworks = obs[:,:,292:317].reshape(80,batch_size,5,5)
    fireworks_empty_mask = (fireworks.sum(-1, keepdim=True) == 0.0)
    fireworks = torch.cat([fireworks, fireworks_empty_mask.float()], -1)
    fireworks = (fireworks == 1).nonzero(as_tuple=True)[3].reshape(80,batch_size,5)
    for c in range(5):
        fireworks[:,:,c] = 67+5*c+fireworks[:,:,c]

    # 93-101 is info tokens

    info_tokens = 93 + torch.sum(obs[:,:,317:325], -1, dtype = torch.long)

    # 102-105 is life tokens

    life_tokens = 102 + torch.sum(obs[:,:,325:328], -1, dtype = torch.long)

    # 106-145 is last action
        #   106-110 is play
        #   111-115 is discard
        #   116-120 is colour hint
        #   121-125 is rank hint
        #       + 20 for player 2
    # 146-170 is card played/discarded
    # 171-202 is cards affected by hint (2*2*2*2*2 possible hints)
    # 203,204 no-op action

    move_type = obs[1:,:,380:384]
    move_type_empty_mask = (move_type.sum(-1, keepdim=True) == 0.0)
    move_type = torch.cat([move_type, move_type_empty_mask.float()], -1)
    move_type = (move_type == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    move_type = 5*move_type + 106

    which_colour = obs[1:,:,386:391]
    which_rank = obs[1:,:,391:396]
    which_play_disc = obs[1:,:,401:406]

    which_colour_empty_mask = (which_colour.sum(-1, keepdim=True) == 0.0)
    which_colour = torch.cat([which_colour_empty_mask.float(), which_colour], -1)

    which_rank_empty_mask = (which_rank.sum(-1, keepdim=True) == 0.0)
    which_rank = torch.cat([which_rank_empty_mask.float(), which_rank], -1)

    which_play_disc_empty_mask = (which_play_disc.sum(-1, keepdim=True) == 0.0)
    which_play_disc = torch.cat([which_play_disc_empty_mask.float(), which_play_disc], -1)

    which_colour = (which_colour == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    which_rank = (which_rank == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    which_play_disc = (which_play_disc == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)

    move_type += (which_colour + which_rank + which_play_disc - 1)

    which_player = obs[1:,:,378:380]
    which_player_empty_mask = (which_player.sum(-1, keepdim=True) == 0.0)
    which_player = torch.cat([which_player, which_player_empty_mask.float()], -1)
    which_player = (which_player == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    move_type += (20*which_player)

    move_affect = obs[1:,:,406:431]
    move_affect_empty_mask = (move_affect.sum(-1, keepdim=True) == 0.0)
    move_affect = torch.cat([move_affect, move_affect_empty_mask.float()], -1)
    move_affect = (move_affect == 1).nonzero(as_tuple=True)[2].reshape(-1, batch_size)
    move_affect += 146

    move_affect += (obs[1:,:,396:401].matmul(2**torch.arange(5, device=device, dtype=torch.float).flip(0).view(5,1))).reshape(-1, batch_size).type(torch.long)

                                        # 203, 204, 5603, 5604
    move_type = torch.cat([torch.tensor([203 for _ in range(batch_size)], device=device).reshape(1,batch_size), move_type], 0)
    move_affect = torch.cat([torch.tensor([204 for _ in range(batch_size)], device=device).reshape(1,batch_size), move_affect], 0)

    stacked = torch.stack([partner_cards[:,:,0], partner_cards[:,:,1], partner_cards[:,:,2], 
                            partner_cards[:,:,3], partner_cards[:,:,4], decksizes, fireworks[:,:,0],
                            fireworks[:,:,1], fireworks[:,:,2], fireworks[:,:,3], fireworks[:,:,4], 
                            info_tokens, life_tokens, move_type, move_affect] , dim=1)
                            # ck_tokens[:,:,0], ck_tokens[:,:,1],
                            # ck_tokens[:,:,2], ck_tokens[:,:,3], ck_tokens[:,:,4], ck_tokens[:,:,5], ck_tokens[:,:,6],
                            # ck_tokens[:,:,7], ck_tokens[:,:,8], ck_tokens[:,:,9], ck_tokens[:,:,10], ck_tokens[:,:,11],
                            # ck_tokens[:,:,12], ck_tokens[:,:,13], ck_tokens[:,:,14], ck_tokens[:,:,15], ck_tokens[:,:,16],
                            # ck_tokens[:,:,17], ck_tokens[:,:,18], ck_tokens[:,:,19], ck_tokens[:,:,20], ck_tokens[:,:,21],
                            # ck_tokens[:,:,22], ck_tokens[:,:,23], ck_tokens[:,:,24]], dim=1)

    interleaved = torch.flatten(stacked, start_dim = 0, end_dim = 1).transpose(0,1)

    for j in range(batch_size):
        interleaved[j, (gamesteps[j]+1)*15:] = 205 # 205, 5606
   
    return interleaved, trgs


def create_masks(input_seq, target_seq):
    # creates mask with 0s wherever there is padding in the input
    input_pad = 5606
    input_msk = (input_seq != input_pad).unsqueeze(1)

    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)

    return input_msk, nopeak_mask

def parse_args():
    parser = argparse.ArgumentParser(description="train belief model")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="iql")
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
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--warm_up_period", type=float, default=100000, help="Warm Up Period")
    parser.add_argument("--eps", type=float, default=1e-9, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:1")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--eval_epochs", type=int, default=1)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=1000)
    parser.add_argument("--replay_buffer_size", type=int, default=60000)
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    args.load_model = "models/sad-op/M1.pthw"

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
        # args.obs_n_most_recent_own_cards # modification by CHRISTIAN
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
        overwrite["device"] = "cuda:1"
        overwrite["boltzmann_act"] = False
        agent_id = int(args.load_model[args.load_model.index('M')+1:args.load_model.index('.')])
        agent, cfg = utils.load_agent(
            args.load_model,
            overwrite,
            op_agent_id=agent_id,
        )
        agent.log_trajectories = True
        print("CFG: ", cfg)
        assert cfg["num_player"] == args.num_player, "Model num players does not coincide with config num players!"
        print("*****done*****")

    agent.sync_target_with_online()

    if args.load_model and not args.model_is_op:
        print("*****loading pretrained model*****")
        # utils.load_weight(agent.online_net, args.load_model, args.train_device)
        utils.load_weight(agent.online_net, args.load_model, "cpu")
        print("*****done*****")

    agent = agent.to(args.train_device)
    eval_agent = agent.clone(args.train_device, {"vdn": False})

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
  
    context.start()
    
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    print("Success, Done")
    print("=======================")


    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stopwatch = common_utils.Stopwatch()

    obs = torch.zeros((args.batchsize*2, 1200)).type(torch.ShortTensor)
    target = torch.zeros((args.batchsize*2, args.hand_size+2)).type(torch.ShortTensor)

    obs_ = torch.zeros((args.batchsize, 1200)).type(torch.ShortTensor)
    target_ = torch.zeros((args.batchsize, args.hand_size+2)).type(torch.ShortTensor)

    count = 0
    policy = 3
    while True:
        if count % 50 == 0:
            if policy == 13:
                policy = 1 
                count = 0
            args.load_model = "models/sad-op/M"+str(policy)+".pthw"
            policy+=2
            print("*****loading pretrained model*****")
            overwrite = {}
            overwrite["vdn"] = (args.method == "vdn")
            overwrite["device"] = args.act_device
            overwrite["boltzmann_act"] = False
            agent_id = int(args.load_model[args.load_model.index('M')+1:args.load_model.index('.')])
            agent, cfg = utils.load_agent(
                args.load_model,
                overwrite,
                op_agent_id=agent_id,
            )
            agent.log_trajectories = True
            print("CFG: ", cfg)
            assert cfg["num_player"] == args.num_player, "Model num players does not coincide with config num players!"
            print("*****done*****")
            act_group.update_model(agent)
        count += 1

        if len(os.listdir('multi_replays/')) > 500:
            filelist = [f for f in os.listdir('multi_replays/')]
            for f in filelist:
                os.remove(os.path.join('multi_replays', f))
        print("Num batches: " + str(len(os.listdir('multi_replays/'))//2))
        while len(os.listdir('multi_replays/')) > 400 and len(os.listdir('multi_replays/')) % 2 == 0:
            print("Too many files, waiting...")
            time.sleep(1)
        
        obs *= 0
        target *= 0

        for j in range(2):
            obs_ *= 0
            target_ *= 0
            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            obs_[:], target_[:] = get_samples(batch, args)[:]
            obs[j*args.batchsize:(j+1)*args.batchsize, :] = obs_
            target[j*args.batchsize:(j+1)*args.batchsize, :] = target_
            priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
            priority = rela.aggregate_priority(
                priority.cpu(), batch.seq_len.cpu(), args.eta
            )
            replay_buffer.update_priority(priority)

        file_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
        torch.save(obs, "multi_replays/batches_src_"+str(file_id)+".pt")
        torch.save(target, "multi_replays/batches_trg_"+str(file_id)+".pt")

