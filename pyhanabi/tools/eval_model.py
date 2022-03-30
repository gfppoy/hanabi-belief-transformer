# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)

import numpy as np
import torch
import r2d2
import utils
from eval import evaluate


def evaluate_legacy_model(
    weight_files, num_game, seed, bomb, num_run=1, verbose=True
):
    # model_lockers = []
    # greedy_extra = 0
    agents = []
    num_player = len(weight_files)
    assert num_player > 1, "1 weight file per player"

    for weight_file in weight_files:
        if verbose:
            print(
                "evaluating: %s\n\tfor %dx%d games" % (weight_file, num_run, num_game)
            )
        if "sad" in weight_file or "aux" in weight_file:
            sad = True
        else:
            sad = False

        device = "cuda:0"

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]

        agent = r2d2.R2D2Agent(
            False, 3, 0.999, 0.9, device, input_dim, hid_dim, output_dim, 2, 5, False
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,
            sad,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: ", perfect_rate)
    return mean, sem, perfect_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--weight", default=None, type=str, required=True)
    # parser.add_argument("--num_player", default=None, type=int, required=True)
    args = parser.parse_args()

    # assert os.path.exists(args.weight)
    # we are doing self player, all players use the same weight
    # weight_files = [args.weight for _ in range(args.num_player)]


    scores = np.zeros((13,13))
    errors = np.zeros((13,13))
    perfects = np.zeros((13,13))

    for i in range(1,14):
        for j in range(1,i+1):
            weight_files = ["models/sad/sad_2p_"+str(i)+".pthw", "models/sad/sad_2p_"+str(j)+".pthw"]
            mean, sem, perfect_rate = evaluate_legacy_model(weight_files, 1000, 1, 0, num_run=10)
            scores[i-1, j-1] = mean
            errors[i-1, j-1] = sem
            perfects[i-1, j-1] = perfect_rate
            print(str(i)+" and "+str(j))

    np.save('crossplay_score.npy', scores)
    np.save('crossplay_errors.npy', errors)
    np.save('crossplay_perfects.npy', perfects)

    # weight_files = ["models/sad/sad_2p_1.pthw", "models/sad/sad_2p_1.pthw"]

    # fast evaluation for 10k games
    # evaluate_legacy_model(weight_files, 1000, 1, 0, num_run=10)
