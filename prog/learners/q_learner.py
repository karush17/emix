"""Implements the Q learner."""

import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.supmix import SupMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    """Implements the COMA learner.

    Attributes:
        args: experiment arguments.
        n_agents: number of agents.
        n_actions: number of actions.
        max: multi agent controller.
        logger: logger object.
        last_target_update_step: update step of last value.
        log_states_t: state logging at timestep.
        agent_params: parameters of agent network.
        params: agent + mixer params.
        optimiser: policy optimizer.
        mixer: agent Q value mixer.
        supmixer: surprise value mixer.
        target_mixer1: first target mixer network for surpise.
        target_mixer2: second target mixer network for surpise.
        target_supmixer: target surprise value mixer.
    """

    def __init__(self, mac, scheme, logger, args):
        """Initializes the learner object."""
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "emix":
                self.mixer = QMixer(args)
                self.supmixer = SupMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer1 = copy.deepcopy(self.mixer)
            self.target_mixer2 = copy.deepcopy(self.mixer)
            self.target_supmixer = copy.deepcopy(self.supmixer)

        self.optimiser = RMSprop(params=self.params,
                                 lr=args.lr,
                                 alpha=args.optim_alpha,
                                 eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        """Trains the Q learner agent.
        
        Args:
            batch: batch of data samples.
            t_env: env timestep.
            episode_num: number of episodes.
        """
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1],
                                        dim=3,
                                        index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)

        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals,
                                             batch["state"][:, :-1])
            target_max_qvals1 = self.target_mixer1(target_max_qvals,
                                                   batch["state"][:, 1:])
            target_max_qvals2 = self.target_mixer2(target_max_qvals,
                                                   batch["state"][:, 1:])

        min_target_vals = th.min(target_max_qvals1, target_max_qvals2)
        if self.args.mixer == "emix":
            surp_vals = self.supmixer(chosen_action_qvals,
                                      batch["state"][:, :-1],
                                      th.std(batch["state"][:, :-1], dim=2).unsqueeze(2))
            target_surp_vals = self.target_supmixer(min_target_vals,
                                                    batch["state"][:, 1:],
                                                    th.std(batch["state"][:, 1:], dim=2).unsqueeze(2))

        temp = self.args.temp
        targets = rewards + temp*th.logsumexp(target_surp_vals,
                                              dim=1)
        + self.args.gamma * (1 - terminated) * min_target_vals

        td_error = (temp*th.logsumexp(surp_vals, dim=1)
                    + chosen_action_qvals - targets)
        surprise_vals = th.mean(temp*th.logsumexp(surp_vals, dim=1))
        surprise_ratio = th.mean(temp*th.logsumexp(surp_vals, dim=1)
                                 - temp*th.logsumexp(target_surp_vals, dim=1))

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()


        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("surprise_vals", surprise_vals.item(), t_env)
            self.logger.log_stat("surprise_ratio", surprise_ratio.item(), t_env)
            self.logger.log_stat("td_error_abs",
                                 (masked_td_error.abs().sum().item()/mask_elems),
                                 t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item()
                                 /(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean",
                                 (targets * mask).sum().item()
                                 /(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        """Updates target networks."""
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer1.load_state_dict(self.mixer.state_dict())
            self.target_mixer2.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        """Moves agent and critic to cuda."""
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.supmixer.cuda()
            self.target_mixer1.cuda()
            self.target_mixer2.cuda()
            self.target_supmixer.cuda()

    def save_models(self, path):
        """Saves models."""
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        """Loads models."""
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                               map_location=lambda storage,
                                               loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path),
                                               map_location=lambda storage,
                                               loc: storage))
