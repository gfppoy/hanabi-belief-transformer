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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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

    _, trg_mask = create_masks(obs, target[:, :-1])

    losses = []
    target = target.to(device)
    obs = obs.to(device)
    # ck = ck.to(device)
    trg_mask = trg_mask.to(device)

    preds = model(obs, ck, target[:, :-1], None, trg_mask)#.to("cpu")

    for j in range(5):
        loss = F.cross_entropy(preds[:,j,:].view(-1, preds.size(-1)), target[:,j+1].contiguous().view(-1), ignore_index = 5606)
        total_losses[j] += loss.item()
        losses.append(loss)

    return total_losses, losses


def create_masks(input_seq, target_seq):
    # creates mask with 0s wherever there is padding in the input
    input_pad = 5606
    input_msk = (input_seq != input_pad).unsqueeze(1)

    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)

    return input_msk, nopeak_mask
        
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
    parser.add_argument("--lr", type=float, default=0.00025, help="Learning rate")
    parser.add_argument("--warm_up_period", type=float, default=120000, help="Warm Up Period")
    parser.add_argument("--eps", type=float, default=1e-9, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:0")
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args = parse_args()

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

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stopwatch = common_utils.Stopwatch()

    ##############
    # Create belief model
    #############

    model = get_model(206, 28, 256, args.N, args.num_heads).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.998), eps=args.eps)

    ##############
    # END Create belief model
    #############

    obs = torch.zeros((args.batchsize, 1200)).type(torch.LongTensor)
    target = torch.zeros((args.batchsize, args.hand_size+2)).type(torch.LongTensor)

    obses = torch.zeros((args.batchsize*2, 1200)).type(torch.LongTensor)
    targets = torch.zeros((args.batchsize*2, args.hand_size+2)).type(torch.LongTensor)

    total_losses = np.zeros(5)
    print_every = 50
    save_loss_every = 250
    save_entr_vs_timestep_every = 3000
    running_losses = [[] for _ in range(5)]

    best_model_score = 100

    step_num = 1

    start = time.time()
    temp = start

    for epoch in range(args.num_epoch):
        print("beginning of epoch: " + str(epoch+1))
        print(common_utils.get_mem_usage())
        stopwatch.reset()
        loss_lst_train = None

        for batch_idx in range(args.epoch_len):
            if epoch == -1: # DBG
                break

            if batch_idx % 2 == 0:
                obses *= 0
                targets *= 0
                while len(os.listdir('multi_replays/')) < 8:
                    print("Waiting for more samples...")
                    time.sleep(1)

                while True:
                    while len(os.listdir('multi_replays/')) < 8:
                        print("Waiting for more samples...")
                        time.sleep(1)
                    candidate = random.choice(os.listdir("multi_replays/"))
                    while not (candidate.startswith("batches_") and candidate.endswith(".pt")):
                        while len(os.listdir('multi_replays/')) < 8:
                            print("Waiting for more samples...")
                            time.sleep(1)
                        candidate = random.choice(os.listdir("multi_replays/"))
                    try:
                        obses = torch.load("multi_replays/"+candidate.replace("trg","src").replace("ck","src"))
                        targets = torch.load("multi_replays/"+candidate.replace("src","trg").replace("ck","trg"))

                        os.remove("multi_replays/"+candidate.replace("trg","src").replace("ck","src"))
                        os.remove("multi_replays/"+candidate.replace("src","trg").replace("ck","trg"))
                        break

                    except:
                        try:
                            os.remove("multi_replays/"+candidate.replace("trg","src").replace("ck","src"))
                        except:
                            pass
                        try:
                            os.remove("multi_replays/"+candidate.replace("src","trg").replace("ck","trg"))
                        except:
                            pass

            stopwatch.time("sync and updating")

            if args.belief_type == "public":
                obs = batch.obs["priv_s"][:, :, -feature_size:] if args.num_player < 5 else batch.obs["priv_s"][:, :, 0, -feature_size:]
                obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1, 1, args.num_player, 1).float()], -1)
                obs = obs[:torch.max(batch.seq_len).long().item()]
                target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size, 25) # have to add empty card dimension to target!
                target_empty_mask = (target.sum(-1, keepdim=True)==0.0)
                target = torch.cat([target, target_empty_mask.float()], -1)
                target = target[:torch.max(batch.seq_len).long().item()]
            elif args.belief_type == "own_hand":
                obs *= 0
                target *= 0
                obs[:] = obses[(batch_idx % 2)*args.batchsize:(1 + batch_idx % 2)*args.batchsize,:]
                target[:] = targets[(batch_idx % 2)*args.batchsize:(1 + batch_idx % 2)*args.batchsize,:]
                stopwatch.time("sample data")

            else:
                assert False, "Unknown belief type: {}".format(args.belief_type)

            total_losses, step_num = belief_run(model,obs.type(torch.LongTensor),None,target.type(torch.LongTensor),total_losses,args,optim,step_num,stopwatch,device,'train')
            torch.cuda.synchronize()

            if (batch_idx + 1) % print_every == 0:
                loss_avgs = [total_losses[j] / print_every for j in range(5)]
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
                epoch + 1, batch_idx + 1, 0.2*sum(loss_avgs), time.time() - temp,
                print_every))
                for j in range(5):
                    running_losses[j].append(loss_avgs[j])
                    # running_tests[j].append(loss_test_avgs[j])
                if 0.2*sum(loss_avgs) < best_model_score:
                    torch.save(model.state_dict(),"/pyhanabi/saves_while_training/model$_multi6.pth")
                    best_model_score = 0.2*sum(loss_avgs)

                total_losses = np.zeros(5)
                total_tests  = np.zeros(5)
                temp = time.time()

            if (batch_idx + 1) % (save_loss_every) == 0:
                np.save(os.path.join('saves_while_training', 'train_multi6.npy'), np.array(running_losses))

        stopwatch.summary()

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: " + str(epoch+1))
        torch.cuda.synchronize()

        print("==========")
