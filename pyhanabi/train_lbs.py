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

from create import create_envs, create_threads, ActGroup
from eval import evaluate
import common_utils
import rela
import r2d2_lbs as r2d2
import utils

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
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_obs", type=int, default=0)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--pred_weight", type=float, default=0)
    parser.add_argument("--num_eps", type=int, default=80)

    parser.add_argument("--load_model", type=str, default="models/aux/aux_2p_13.pthw")

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--eta", type=float, default=0.9, help="eta for aggregate priority")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=1)
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--hand_size", type=int, default=5)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--eval_epochs", type=int, default=1)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=5000)
    parser.add_argument("--replay_buffer_size", type=int, default=6000)
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
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
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
    assert args.load_model != "", "You need to load a model in train LBS mode!"
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_eps
    )
    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))

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
    elif args.load_model != "" and args.model_is_op:
        agent = load_op_model(args.method, args.idx, args.idx, args.train_device)[0]

    # else:
    #     agent = r2d2.R2D2Agent(
    #         (args.method == "vdn"),
    #         args.multi_step,
    #         args.gamma,
    #         args.eta,
    #         args.train_device,
    #         feature_size, # if not args.use_softmax_policy else feature_size + 1,
    #         args.rnn_hid_dim,
    #         games[0].num_action(),
    #         args.num_lstm_layer,
    #         args.hand_size,
    #         args.no_replay_buffer_priority,  # uniform priority
    #         args.use_softmax_policy,
    #         args.q_variant
    #     )

    agent.sync_target_with_online()

    if args.load_model and not args.model_is_op:
        print("*****loading pretrained model*****")
        utils.load_weight(agent.online_net, args.load_model, args.train_device)
        print("*****done*****")

    agent = agent.to(args.train_device)
    #optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(agent)
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

    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()

    ##############
    # Create LBS model
    #############

    mdl = r2d2.LBSNetLSTM #if args.use_autoregressive_belief else r2d2.LBSNetFF

    lbs_net = mdl(device=args.train_device,
                  in_dim=feature_size + args.num_player,# +1 for the actions so far
                  hid_dim=args.rnn_hid_dim,
                  out_dim=args.hand_size*args.card_feature_dim, # ignore if public belief
                  num_lstm_layer=args.num_lstm_layer,
                  num_fc_layer=1,
                  card_feature_dim=args.card_feature_dim,
                  n_cards=args.hand_size if args.belief_type == "own_hand" else args.hand_size*args.num_player,
                  use_ff=True).to(args.train_device)

    optim = torch.optim.Adam(lbs_net.parameters(), lr=args.lr, eps=args.eps)
    ##############
    # END Create LBS model
    #############

    loss_lst_train_lst = []
    loss_lst_lst = []
    for epoch in range(-1, args.num_epoch):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()
        loss_lst_train = None
        for batch_idx in range(args.epoch_len):
            if epoch == -1: # DBG
                break
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                act_group.update_model(agent)

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch.time("sample data")

            obs = None
            target = None
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
                obs = batch.obs["priv_s"]
                # if args.num_player < 5:
                #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-1).float()], -1)
                # else:
                obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1, 1, args.num_player, 1).float()], -1)
                obs = obs[:torch.max(batch.seq_len).long().item()]
                target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size, 25) # have to add empty card dimension to target!
                target_empty_mask = (target.sum(-1, keepdim=True)==0.0)
                target = torch.cat([target, target_empty_mask.float()], -1)
                target = target[:torch.max(batch.seq_len).long().item()]
            else:
                assert False, "Unknown belief type: {}".format(args.belief_type)
            loss_lst_train, priority = agent.loss_lbs(lbs_net, obs, target, stat) # , args.pred_weight,
            loss_lst_train_lst.append( loss_lst_train.detach().cpu())

            # MOD2: ReplayBuffer aggregate_priority not needed
            priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
            priority = rela.aggregate_priority(
                priority.cpu(), batch.seq_len.cpu(), args.eta
            )
            loss = (loss_lst_train).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            # MOD1: ReplayBuffer update_priotity is not needed
            replay_buffer.update_priority(priority)
            stopwatch.time("updating priority")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        tachometer.lap(
            act_group.actors, replay_buffer, args.epoch_len * args.batchsize, count_factor
        )
        stopwatch.summary()
        stat.summary(epoch)

        context.pause()

        eval_seed = (9917 + epoch * 999999) % 7777777
        eval_agent.load_state_dict(agent.state_dict())

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

        print("Evaluating policy performance...")
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
        d = evaluate(
            [eval_agent for _ in range(args.num_player)],
            1000,
            eval_seed,
            args.eval_bomb,
            0,  # explore eps
            args.sad,
            args.obs_n_most_recent_own_cards,
            hand_size=args.hand_size,
            use_softmax_policy=args.use_softmax_policy,
            eval_log_betas=[float(b) for b in args.eval_log_betas.split(",")]
        )
        torch.cuda.synchronize()

        print("Evaluating belief model performance...")
        stopwatch.time("sync and updating")

        batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
        stopwatch.time("sample data")

        obs = None
        target = None
        if args.belief_type == "public":#
            assert False, "public belief not currently supported!"
            # if args.num_player < 5:
            #     obs = batch.obs["priv_s"][:, :, -feature_size:]
            #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-1).float()], -1)
            #     target = batch.obs["own_hand"].view(batch.obs["own_hand"].shape[0],
            #                                         batch.obs["own_hand"].shape[1],
            #                                         args.hand_size * args.num_player * args.card_feature_dim)
            # else:
            obs = batch.obs["priv_s"][:, :, 0, -feature_size:]
            obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1, 1, args.num_player, 1).float()], -1)
            # TODO: This does not work for public belief!
            target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size,
                                                25)  # have to add empty card dimension to target!
            target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
            target = torch.cat([target, target_empty_mask.float()], -1)
            obs = obs[:torch.max(batch.seq_len).long().item()]
            target = target[:torch.max(batch.seq_len).long().item()]
        elif args.belief_type == "own_hand":
            obs = batch.obs["priv_s"]
            # if args.num_player < 5:
            #     obs = torch.cat([obs, batch.action["a"].unsqueeze(-1).float()], -1)
            # else:
            obs = torch.cat([obs, batch.action["a"].unsqueeze(-2).repeat(1,1,args.num_player,1).float()], -1)
            obs = obs[:torch.max(batch.seq_len).long().item()]
            target = batch.obs["own_hand"].view(*batch.obs["own_hand"].shape[:3], args.hand_size,
                                                25)  # have to add empty card dimension to target!
            target_empty_mask = (target.sum(-1, keepdim=True) == 0.0)
            target = torch.cat([target, target_empty_mask.float()], -1)
            target = target[:torch.max(batch.seq_len).long().item()]
        else:
            assert False, "Unknown belief type: {}".format(args.belief_type)
        loss_lst, priority = agent.loss_lbs(lbs_net, obs, target, stat, eval=True)  # , args.pred_weight,
        loss_lst_lst.append(loss_lst.detach().cpu())

        # MOD2: ReplayBuffer aggregate_priority not needed
        priority = torch.zeros(batch.obs["priv_s"].shape[:2], device=batch.obs["priv_s"].device) + 1.0
        priority = rela.aggregate_priority(
            priority.cpu(), batch.seq_len.cpu(), args.eta
        )

        if loss_lst_train is not None:
            loss_train = loss_lst_train.mean().detach()
            print("CrossEntropyLoss: ", loss_train)

        loss = loss_lst.mean().detach()
        print("Category Loss: ", loss)

        replay_buffer.update_priority(priority)
        torch.cuda.synchronize()

        if epoch > 0 and epoch % 50 == 0:
            force_save_name = "model_epoch%d" % epoch
        else:
            force_save_name = None

        if epoch == -1 or epoch%args.eval_epochs == 0:
            print("Saving belief model...")
            torch.save({"LBSBeliefNet": lbs_net.state_dict()}, os.path.join(args.save_dir, "lbs_model_ep{}.pthw".format(epoch)))
            print("Saving cross entropy graph...")
            import matplotlib.pyplot as plt

            loss_lst = loss_lst.detach().view(loss_lst.shape[0], -1).mean(-1).cpu()
            print("crossentropy data (val - hard):")
            print(loss_lst)

            # Plot average cross-entropy per step
            plt.plot(list(range(len(loss_lst))), loss_lst, label="val")
            if loss_lst_train is not None:
                loss_lst_train = loss_lst_train.detach().view(loss_lst_train.shape[0], -1).mean(-1).cpu()
                plt.plot(list(range(len(loss_lst_train))), loss_lst_train, label="train")
                print("crossentropy data (train - soft):")
                print(loss_lst_train)

            plt.xlabel('steps')
            plt.ylabel('cross entropy (whole hand)')
            plt.title("Epoch {}".format(epoch))
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, "lbs_model_ep{}.png".format(epoch)), bbox_inches='tight')
            plt.clf()

            # Train loss over time plot
            if loss_lst_train_lst:
                #lt = torch.stack(loss_lst_train_lst, 0)
                # lt = pad_sequence(loss_lst_train_lst, batch_first=True).permute(1,0,2,3)
                # lt = lt.view(lt.shape[0], -1).mean(-1, keepdim=True)
                _lt = [ _l.mean() for _l in loss_lst_train_lst]
                plt.plot(list(range(len(loss_lst_train_lst))), _lt, label="Train cross-entropy loss")
                # for i, _lt in enumerate(lt):
                #    plt.plot(list(range(i+1)), _lt, label="game steps {}".format(i))
                plt.xlabel('training episodes')
                plt.ylabel('cross entropy (whole hand)')
                plt.title("Epoch {}".format(epoch))
                plt.legend()
                plt.savefig(os.path.join(args.save_dir, "lbs_model_trainloss_ep{}.png".format(epoch)), bbox_inches='tight')
                plt.clf()

            # Eval loss over time plot
            if loss_lst_lst:
                #lt = pad_sequence(loss_lst_lst, batch_first=True)
                # lt = pad_sequence(loss_lst_lst, batch_first=True).permute(1, 0, 2, 3)
                # lt = lt.view(lt.shape[0], -1).mean(-1, keepdim=True)
                _lt = [ _l.mean() for _l in loss_lst_lst]
                plt.plot(list(range(len(loss_lst_lst))), _lt, label="Eval category loss")
                plt.xlabel('training episodes')
                plt.ylabel('cross entropy (whole hand)')
                plt.title("Epoch {}".format(epoch))
                plt.legend()
                plt.savefig(os.path.join(args.save_dir, "lbs_model_valloss_ep{}.png".format(epoch)), bbox_inches='tight')
                plt.clf()

            #score, score_std, perfect, *_ = d
            scores_mean = d["scores_mean"]
            scores_std = d["scores_std"]
            fraction_perfect = d["fraction_perfect"]
            #model_saved = saver.save(
            #    None, agent.online_net.state_dict(), scores_mean, force_save_name=force_save_name
            #)
            print(
                "epoch %d, eval score: %.4f, eval score std: %.4f, fraction perfect: %.2f"
                % (epoch, scores_mean, scores_std, fraction_perfect * 100)
            )

        context.resume()
        print("==========")
