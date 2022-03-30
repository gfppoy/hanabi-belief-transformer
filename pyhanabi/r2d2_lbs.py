# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict
import common_utils


class R2D2Net(torch.jit.ScriptModule):
    __constants__ = [
        "hid_dim",
        "out_dim",
        "num_lstm_layer",
        "hand_size",
        "skip_connect",
    ]

    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        num_fc_layer,
        skip_connect,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_fc_layer = num_fc_layer
        self.num_lstm_layer = num_lstm_layer
        self.hand_size = hand_size
        self.skip_connect = skip_connect

        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_fc_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred = nn.Linear(self.hid_dim, self.hand_size * 3)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self, priv_s: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        #if priv_s.dim() > 2:
        #    priv_s = priv_s.view(-1, priv_s.shape[-1])
        #     #DBG FOR OP ONLY!
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % s.dim()

        priv_s = priv_s.unsqueeze(0)
        x = self.net(priv_s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        if self.skip_connect:
            o = o + x
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.net(priv_s)
        if len(hid) == 0:
            o, (h, c) = self.lstm(x)
        else:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self._duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)

        return qa, greedy_action, q, o

    @torch.jit.script_method
    def _duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    def cross_entropy(self, net, lstm_o, target_p, hand_slot_mask, seq_len):
        # target_p: [seq_len, batch, num_player, 5, 3]
        # hand_slot_mask: [seq_len, batch, num_player, 5]
        logit = net(lstm_o).view(target_p.size())
        q = nn.functional.softmax(logit, -1)
        logq = nn.functional.log_softmax(logit, -1)
        plogq = (target_p * logq).sum(-1)
        xent = -(plogq * hand_slot_mask).sum(-1) / hand_slot_mask.sum(-1).clamp(
            min=1e-6
        )

        if xent.dim() == 3:
            # [seq, batch, num_player]
            xent = xent.mean(2)

        # save before sum out
        seq_xent = xent
        xent = xent.sum(0)
        assert xent.size() == seq_len.size()
        avg_xent = (xent / seq_len).mean().item()
        return xent, avg_xent, q, seq_xent.detach()

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return self.cross_entropy(self.pred, lstm_o, target, hand_slot_mask, seq_len)

class LBSNetFF(torch.jit.ScriptModule):

    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        num_fc_layer,
        card_feature_dim,
        n_cards,
        use_ff=True
    ):
        super().__init__()

        self.card_feature_dim = card_feature_dim
        self.n_cards = n_cards
        self.use_ff = use_ff
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_fc_layer = num_fc_layer
        self.num_lstm_layer = num_lstm_layer

        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_fc_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.ffnet = nn.Sequential(*ff_layers)

        self.lstm_t = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm_t.flatten_parameters()
        self.pred_ff1 = nn.Sequential(*[nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU()])
        self.pred_ff2 = nn.Linear(self.hid_dim, self.n_cards*self.card_feature_dim)
        pass

    @torch.jit.script_method
    def forward(
        self,
        player_public_private_obs: torch.Tensor,
        eval: bool=False,
    ) -> torch.Tensor:

        x = self.ffnet(player_public_private_obs)
        # print("XE:", x.shape)
        y = x.view(x.shape[0], -1, x.shape[-1])
        xE, _ = self.lstm_t(y)
        y = self.pred_ff1(xE)
        cards_pred = self.pred_ff2(y)
        return cards_pred.view(x.shape[0],
                               x.shape[1],
                               x.shape[2],
                               self.n_cards,
                               self.card_feature_dim)


class LBSNetLSTM(torch.jit.ScriptModule):

    def __init__(
            self,
            device,
            in_dim,
            hid_dim,
            out_dim,
            num_lstm_layer,
            num_fc_layer,
            card_feature_dim,
            n_cards,
            use_ff=True
    ):
        super().__init__()

        self.card_feature_dim = card_feature_dim
        self.n_cards = n_cards
        self.use_ff = use_ff
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        # self.out_dim = out_dim
        self.num_fc_layer = num_fc_layer
        self.num_lstm_layer = num_lstm_layer

        ff_layers = [nn.Linear(self.in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_fc_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.net = nn.Sequential(*ff_layers)

        self.lstm_t = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm_t.flatten_parameters()

        self.lstm_c = nn.LSTMCell(
            self.card_feature_dim, self.hid_dim
        ).to(device)
        self.pred = nn.Linear(self.hid_dim, self.card_feature_dim)

    @torch.jit.script_method
    def get_context_rep(self, context):
        context_ = self.net(context.view(context.shape[0], -1, context.shape[-1]))
        context_, _ = self.lstm_t(context_) # xE: t x b x v
        context_ = context_.view(-1, context_.shape[-1])  # xA: (t*b) x v
        return context_

    @torch.jit.script_method
    def sample(self, context_rep, n_samples):
        context_rep_repeat = context_rep.unsqueeze(0).repeat(n_samples, 1, 1).view(-1, context_rep.shape[-1])
        next_card = torch.zeros((context_rep_repeat.shape[0], self.card_feature_dim), device=context_rep.device)
        hx = context_rep_repeat
        cx = context_rep_repeat
        cards_arr = []
        for c in range(self.n_cards):
            hx, cx = self.lstm_c(next_card, (hx, cx))
            next_card_prob = torch.nn.functional.softmax(self.pred(hx), dim=-1)
            next_card = torch.multinomial(next_card_prob, 1)
            next_card = next_card_prob.clone().zero_().scatter_(-1, next_card.long(), 1)
            cards_arr.append(next_card)
        cards = torch.stack(cards_arr, 1)
        return cards

    @torch.jit.script_method
    def forward(
            self,
            inp: torch.Tensor,
            target: torch.Tensor,
            eval: bool=False,
            get_model_entropy: bool=False
            #get_selected_card_prob: bool=False,
            #get_prob_instead_of_cards: bool=False
    ) -> torch.Tensor:

        inp_shape = inp.shape
        inp = self.net(inp.view(inp.shape[0], -1, inp.shape[-1]))
        inp, _ = self.lstm_t(inp) # xE: t x b x v
        inp = inp.view(-1, inp.shape[-1])  # xA: (t*b) x v

        oH = torch.tensor(0)
        if not (eval and len(target.shape) == 0):
            oH = target.view(-1, target.shape[-2], target.shape[-1])

        next_card = torch.zeros((inp.shape[0], self.card_feature_dim), device=inp.device)
        hx = inp
        cx = inp

        output_arr = []
        prob_arr = []
        for c in range(self.n_cards):
            hx, cx = self.lstm_c(next_card, (hx, cx))
            next_card_prob = torch.nn.functional.softmax(self.pred(hx), dim=-1)

            if eval and len(target.shape) == 0: # sample model
                next_card = torch.multinomial(next_card_prob, 1)
                next_card = next_card_prob.clone().zero_().scatter_(-1, next_card.long(), 1)
                if get_model_entropy:
                    next_card_argmax = torch.argmax(next_card, -1, keepdim=True)
                    prob_gather = next_card_prob.gather(-1, next_card_argmax)
                    loss_ = -torch.log(prob_gather)
                    output_arr.append(loss_)
                else:
                    output_arr.append(next_card)

            elif eval and len(target.shape) != 0: # eval loss mode
                next_card = torch.multinomial(next_card_prob, 1)
                next_card = next_card_prob.clone().zero_().scatter_(-1, next_card.long(), 1)
                #true_card = oH[:, self.card_feature_dim * c: self.card_feature_dim * (c + 1)]
                true_card = oH[:, c]
                # print("tc: ", true_card.shape)
                true_card_argmax = torch.argmax(true_card, -1, keepdim=True)
                # print("tca: ", true_card_argmax.shape)
                prob_gather = next_card.gather(-1, true_card_argmax)
                # print("nc:", next_card.shape)
                loss_ = prob_gather
                #empty_card_mask = ~(true_card.sum(-1, keepdim=True).to(torch.bool))
                #loss_.masked_fill_(empty_card_mask, 1.0)
                output_arr.append(loss_)

            else: # train loss mode
                # true_card = oH[:, self.card_feature_dim * c: self.card_feature_dim * (c + 1)]
                true_card = oH[:, c]
                next_card = true_card
                true_card_argmax = torch.argmax(true_card, -1, keepdim=True)
                # do not mask empty card slot - empty slots are public info, so just make sure in the evaluation that
                # you do not include empty slots in the metrics
                #print("ncp: ", next_card_prob.shape)
                #print("tca: ", true_card_argmax.shape)
                prob_gather = next_card_prob.gather(-1, true_card_argmax)
                loss_ = -torch.log(prob_gather)
                #empty_card_mask = ~(true_card.sum(-1, keepdim=True).to(torch.bool))
                #loss_.masked_fill_(empty_card_mask, 0.0)
                output_arr.append(loss_) # add only the logit of the true card

            #if get_selected_card_prob:
            #    prob_arr.append(next_card_prob[next_card.to(torch.bool)])

        if eval and len(target.shape) == 0:
            output_stack = torch.stack(output_arr, -2)
            # if get_selected_card_prob:
            #     print(prob_arr[0].shape)
            #     prob_stack = torch.stack(prob_arr, -1).unsqueeze(-1)
            #     # print(prob_stack.shape)
            #     # print(output_stack.shape)
            #     return torch.cat([output_stack, prob_stack], -1)
            # else:
            return output_stack
        else:
            output_stack = torch.stack(output_arr, -1)
            output_stack = output_stack.view(inp_shape[0],
                                             inp_shape[1],
                                             inp_shape[2],
                                             len(output_arr))
            # if get_selected_card_prob:
            #     prob_stack = torch.stack(prob_arr, -1)
            #     return torch.cat([output_stack, prob_stack], -1)
            # else:
            return output_stack


class StatisticianNet(torch.jit.ScriptModule):

    def __init__(
            self,
            device,
            in_dim,
            hid_dim,
            context_dim,
            num_fc_layer,
            card_feature_dim,
            n_cards,
            use_ff=True
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.context_dim = context_dim
        self.num_fc_layer = num_fc_layer
        self.card_feature_dim = card_feature_dim
        self.n_cards = n_cards

        ff_layers = [nn.Linear(self.in_dim+1, self.hid_dim), nn.ELU()]
        for i in range(1, self.num_fc_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ELU())
        ff_layers.append(nn.Linear(self.hid_dim, self.context_dim))
        self.net = nn.Sequential(*ff_layers).to(device)
        self.lstm_c = nn.LSTMCell(
            self.card_feature_dim, self.hid_dim
        ).to(device)
        self.pred = nn.Linear(self.hid_dim, self.card_feature_dim).to(device)
        pass

    @torch.jit.script_method
    def get_context_rep(self, context):
        context_shape = list(context.shape)
        context = torch.cat([context, context[:,:,0:1].clone().zero_() + torch.log(context_shape[0])], -1) # attach total number of samples
        context = context.view(-1, context_shape[-1]+1)
        context = self.net(context).view(context_shape[0], context_shape[1], -1)
        context = torch.mean(context, dim=0, keepdim=True)
        context = context.view(-1, context.shape[-1])
        return context

    @torch.jit.script_method
    def sample(self, context_rep, n_samples):
        context_rep_repeat = context_rep.unsqueeze(0).repeat(n_samples, 1, 1).view(-1, context_rep.shape[-1])
        next_card = torch.zeros((context_rep_repeat.shape[0], self.card_feature_dim), device=context_rep.device)
        hx = context_rep_repeat
        cx = context_rep_repeat
        cards_arr = []
        for c in range(self.n_cards):
            hx, cx = self.lstm_c(next_card, (hx, cx))
            next_card_prob = torch.nn.functional.softmax(self.pred(hx), dim=-1)
            next_card = torch.multinomial(next_card_prob, 1)
            next_card = next_card_prob.clone().zero_().scatter_(-1, next_card.long(), 1)
            cards_arr.append(next_card)
        cards = torch.stack(cards_arr, 1)
        return cards

    @torch.jit.script_method
    def forward(
            self,
            context: torch.Tensor,
            targets: torch.Tensor,
            eval: bool=False
    ) -> torch.Tensor:
        # Expects the following dimensions for both contexts:
        # batch x n_context x context
        # for targets:
        # batch x context

        # print("CONTEXT SHAPE 0: ", context.shape)
        # print("TARGETS 0: ", targets.shape)

        # print("RS shape", targets.shape)
        context_shape = list(context.shape)
        context = torch.cat([context, context[:,:,0:1].clone().zero_() + torch.log(context_shape[0])], -1) # attach total number of samples
        context = context.view(-1, context_shape[-1]+1)
        context = self.net(context).view(context_shape[0], context_shape[1], -1)
        context = torch.mean(context, dim=0, keepdim=True)
        # print("CONTEXT SHAPE 1: ", context.shape)
        # print("TARGETS 1: ", targets.shape)
        context = context.repeat(targets.shape[0]//context_shape[1], 1 , 1) # batch_size x n_targets x context_dim
        context = context.view(-1, context.shape[-1]) # pulling together to a single batch size
        # print("MILE 1")

        targets = targets.view(-1, targets.shape[-1]) # pulling together to single batch size

        # print("CONTEXT SHAPE: ", context.shape)
        # print("TARGETS: ", targets.shape)

        next_card = torch.zeros((context.shape[0], self.card_feature_dim), device=context.device)
        hx = context
        cx = context
        # print("MILE 2")
        cards_loss_arr = []
        for c in range(self.n_cards):
            # print("MILE c{}".format(c))
            hx, cx = self.lstm_c(next_card, (hx, cx))
            # print("MILE c{} 0 ".format(c))
            next_card_prob = torch.nn.functional.softmax(self.pred(hx), dim=-1)
            #print("NCP shape: ", next_card_prob.shape)
            #print("NCP1: ", next_card_prob[:10].cpu())
            # print("MILE c{} 1 ".format(c))
            true_card = targets[:, self.card_feature_dim*c: self.card_feature_dim*(c+1)]
            # print("MILE c{} a".format(c))
            if eval:
                next_card = torch.multinomial(next_card_prob, 1)
                next_card = next_card_prob.clone().zero_().scatter_(-1, next_card.long(), 1)

                # print("TS: ", targets.shape)
                # print("TC: ", true_card.shape)
                true_card_argmax = torch.argmax(true_card, -1, keepdim=True)
                prob_gather = next_card.gather(-1, true_card_argmax)
                loss_ = prob_gather
                #empty_card_mask = ~(true_card.sum(-1, keepdim=True).to(torch.bool))
                #loss_.masked_fill_(empty_card_mask, 1.0)
                cards_loss_arr.append(loss_)
                # print("MILE c{} b eval".format(c))
            else:
                next_card = true_card
                true_card_argmax = torch.argmax(true_card, -1, keepdim=True)
                # do not mask empty card slot - empty slots are public info, so just make sure in the evaluation that
                # you do not include empty slots in the loss metrics
                # print("NCP: ", next_card_prob.shape, " TCA:", true_card_argmax.shape)
                #print("NCP:", next_card_prob[:10].cpu())
                prob_gather = next_card_prob.gather(-1, true_card_argmax)
                loss_ = -torch.log(prob_gather)

                #empty_card_mask = ~(true_card.sum(-1, keepdim=True).to(torch.bool))
                #loss_.masked_fill_(empty_card_mask, 0.0)
                cards_loss_arr.append(loss_) # add only the logit of the true card
                # print("MILE c{} b".format(c))

        cards_loss_stack = torch.stack(cards_loss_arr, -1)
        cards_loss_stack = cards_loss_stack.view(context.shape[0],
                                                 -1,
                                                 len(cards_loss_arr))
        return cards_loss_stack

        # # print("FWD")
        # if eval:
        #     return cards_loss_stack
        # else:
        #     return cards_loss_stack


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["vdn", "multi_step", "gamma", "eta", "uniform_priority"]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
        uniform_priority,
        *,
        num_fc_layer=1,
        skip_connect=False,
    ):
        super().__init__()
        self.online_net = R2D2Net(
            device,
            in_dim,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            num_fc_layer,
            skip_connect,
        ).to(device)
        self.target_net = R2D2Net(
            device,
            in_dim,
            hid_dim,
            out_dim,
            num_lstm_layer,
            hand_size,
            num_fc_layer,
            skip_connect,
        ).to(device)
        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.uniform_priority = uniform_priority

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.online_net.num_lstm_layer,
            self.online_net.hand_size,
            self.uniform_priority,
            num_fc_layer=self.online_net.num_fc_layer,
            skip_connect=self.online_net.skip_connect,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, hid)
        # print("NEXTA")
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        obsize, ibsize, num_player = 0, 0, 0


        if self.vdn:
            obsize, ibsize, num_player = obs["priv_s"].size()[:3]
            priv_s = obs["priv_s"].flatten(0, 2)
            legal_move = obs["legal_move"].flatten(0, 2)
            eps = obs["eps"].flatten(0, 2)
        else:
            obsize, ibsize = obs["priv_s"].size()[:2]
            num_player = 1
            priv_s = obs["priv_s"].flatten(0, 1)
            legal_move = obs["legal_move"].flatten(0, 1)
            eps = obs["eps"].flatten(0, 1)

        hid = {
            "h0": obs["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": obs["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }

        # if priv_s.dim() > 2:
        #     priv_s = priv_s.view(-1, priv_s.shape[-1]) # DBG FOR OP
        #     legal_move = legal_move.view(-1, legal_move.shape[-1])
        #     eps = eps.view(-1, eps.shape[-1])
        # print("legal_move shape:", legal_move.shape)
        greedy_action, new_hid = self.greedy_act(priv_s, legal_move, hid)

        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).detach().long()

        if self.vdn:
            action = action.view(obsize, ibsize, num_player)
            greedy_action = greedy_action.view(obsize, ibsize, num_player)
            rand = rand.view(obsize, ibsize, num_player)
        else:
            action = action.view(obsize, ibsize)
            greedy_action = greedy_action.view(obsize, ibsize)
            rand = rand.view(obsize, ibsize)

        hid_shape = (
            obsize,
            ibsize * num_player,
            self.online_net.num_lstm_layer,
            self.online_net.hid_dim,
        )
        h0 = new_hid["h0"].transpose(0, 1).view(*hid_shape)
        c0 = new_hid["c0"].transpose(0, 1).view(*hid_shape)

        reply = {
            "a": action.detach().cpu(),
            "greedy_a": greedy_action.detach().cpu(),
            "h0": h0.contiguous().detach().cpu(),
            "c0": c0.contiguous().detach().cpu(),
        }
        return reply

    @torch.jit.script_method
    def compute_priority(
        self, input_: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        compute priority for one batch
        """
        if self.uniform_priority:
            return {"priority": torch.ones_like(input_["reward"]).detach().cpu()}

        obsize, ibsize, num_player = 0, 0, 0
        flatten_end = 0
        if self.vdn:
            obsize, ibsize, num_player = input_["priv_s"].size()[:3]
            flatten_end = 2
        else:
            obsize, ibsize = input_["priv_s"].size()[:2]
            num_player = 1
            flatten_end = 1

        priv_s = input_["priv_s"].flatten(0, flatten_end)
        legal_move = input_["legal_move"].flatten(0, flatten_end)
        online_a = input_["a"].flatten(0, flatten_end)

        next_priv_s = input_["next_priv_s"].flatten(0, flatten_end)
        next_legal_move = input_["next_legal_move"].flatten(0, flatten_end)
        temperature = input_["temperature"].flatten(0, flatten_end)

        hid = {
            "h0": input_["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        next_hid = {
            "h0": input_["next_h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": input_["next_c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }
        reward = input_["reward"].flatten(0, 1)
        bootstrap = input_["bootstrap"].flatten(0, 1)

        online_qa = self.online_net(priv_s, legal_move, online_a, hid)[0]
        next_a, _ = self.greedy_act(next_priv_s, next_legal_move, next_hid)
        target_qa, _, _, _ = self.target_net(
            next_priv_s, next_legal_move, next_a, next_hid,
        )

        bsize = obsize * ibsize
        if self.vdn:
            # sum over action & player
            online_qa = online_qa.view(bsize, num_player).sum(1)
            target_qa = target_qa.view(bsize, num_player).sum(1)

        assert reward.size() == bootstrap.size()
        assert reward.size() == target_qa.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        priority = (target - online_qa).abs()
        priority = priority.view(obsize, ibsize).detach().cpu()
        return {"priority": priority}

    ############# python only functions #############
    def flat_4d(self, data):
        """
        rnn_hid: [num_layer, batch, num_player, dim] -> [num_player, batch, dim]
        seq_obs: [seq_len, batch, num_player, dim] -> [seq_len, batch, dim]
        """
        bsize = 0
        num_player = 0
        for k, v in data.items():
            if num_player == 0:
                bsize, num_player = v.size()[1:3]

            if v.dim() == 4:
                d0, d1, d2, d3 = v.size()
                data[k] = v.view(d0, d1 * d2, d3)
            elif v.dim() == 3:
                d0, d1, d2 = v.size()
                data[k] = v.view(d0, d1 * d2)
        return bsize, num_player

    def td_error(self, obs, hid, action, reward, terminal, bootstrap, seq_len, stat):
        max_seq_len = obs["priv_s"].size(0)

        bsize, num_player = 0, 1
        if self.vdn:
            bsize, num_player = self.flat_4d(obs)
            self.flat_4d(action)

        priv_s = obs["priv_s"]
        legal_move = obs["legal_move"]
        action = action["a"]

        hid = {}

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qa, greedy_a, _, lstm_o = self.online_net(
            priv_s, legal_move, action, hid
        )

        with torch.no_grad():
            target_qa, _, _, _ = self.target_net(priv_s, legal_move, greedy_a, hid)
            # assert target_q.size() == pa.size()
            # target_qe = (pa * target_q).sum(-1).detach()
            assert online_qa.size() == target_qa.size()

        if self.vdn:
            online_qa = online_qa.view(max_seq_len, bsize, num_player).sum(-1)
            target_qa = target_qa.view(max_seq_len, bsize, num_player).sum(-1)
            lstm_o = lstm_o.view(max_seq_len, bsize, num_player, -1)

        terminal = terminal.float()
        bootstrap = bootstrap.float()

        errs = []
        target_qa = torch.cat(
            [target_qa[self.multi_step :], target_qa[: self.multi_step]], 0
        )
        target_qa[-self.multi_step :] = 0

        assert target_qa.size() == reward.size()
        target = reward + bootstrap * (self.gamma ** self.multi_step) * target_qa
        mask = torch.arange(0, max_seq_len, device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        err = (target.detach() - online_qa) * mask
        return err, lstm_o

    def aux_task_iql(self, lstm_o, hand, seq_len, rl_loss_size, stat):
        seq_size, bsize, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(3)
        pred_loss1, avg_xent1, _, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def aux_task_vdn(self, lstm_o, hand, t, seq_len, rl_loss_size, stat):
        """1st and 2nd order aux task used in VDN"""
        seq_size, bsize, num_player, _ = hand.size()
        own_hand = hand.view(seq_size, bsize, num_player, self.online_net.hand_size, 3)
        own_hand_slot_mask = own_hand.sum(4)
        pred_loss1, avg_xent1, belief1, _ = self.online_net.pred_loss_1st(
            lstm_o, own_hand, own_hand_slot_mask, seq_len
        )
        assert pred_loss1.size() == rl_loss_size

        rotate = [num_player - 1]
        rotate.extend(list(range(num_player - 1)))
        partner_hand = own_hand[:, :, rotate, :, :]
        partner_hand_slot_mask = partner_hand.sum(4)
        partner_belief1 = belief1[:, :, rotate, :, :].detach()

        stat["aux1"].feed(avg_xent1)
        return pred_loss1

    def loss(self, batch, pred_weight, stat):

        err, lstm_o = self.td_error(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
            stat,
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

        priority = err.abs()

        if pred_weight > 0:
            if self.vdn:
                pred_loss1 = self.aux_task_vdn(
                    lstm_o,
                    batch.obs["own_hand"],
                    batch.obs["temperature"],
                    batch.seq_len,
                    rl_loss.size(),
                    stat,
                )
                loss = rl_loss + pred_weight * pred_loss1
            else:
                pred_loss = self.aux_task_iql(
                    lstm_o, batch.obs["own_hand"], batch.seq_len, rl_loss.size(), stat,
                )
                loss = rl_loss + pred_weight * pred_loss
        else:
            loss = rl_loss
        return loss, priority
    
    def loss_lbs(self, net, obs, target, stat, eval=False):
        # MODIFICATION: This is the LBS loss, not reinforcement learning is being done ever!
        priority = target.squeeze() ** 0
        output_lst = net(obs, target, eval=eval)
        return output_lst, priority

    def loss_statistician(self, net, D_context, D_holdout, eval=False):
        # MODIFICATION: This is the Statistician loss
        out = net(D_context, D_holdout, eval=eval)
        return out

