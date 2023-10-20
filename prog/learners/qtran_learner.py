"""Implements the QTRAN learner."""

import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qtran import QTranBase
import torch as th
from torch.optim import RMSprop, Adam


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
        target_mixer: target mixer network for surpise.
        target_mac: target multi agent controller.
    """

    def __init__(self, mac, scheme, logger, args):
        """Initializes the learner object."""
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer == "qtran_base":
            self.mixer = QTranBase(args)
        elif args.mixer == "qtran_alt":
            raise Exception("Not implemented here!")

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

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
        mac_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size,
                                                      self.args.n_agents,
                                                      batch.max_seq_length,
                                                      -1).transpose(1,2)

        chosen_action_qvals = th.gather(mac_out[:, :-1],
                                        dim=3,
                                        index=actions).squeeze(3)

        target_mac_out = []
        target_mac_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(self.target_mac.hidden_states)

        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
        target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size,
                                                                    self.args.n_agents,
                                                                    batch.max_seq_length,
                                                                    -1).transpose(1,2)

        target_mac_out[avail_actions[:, :] == 0] = -9999999
        mac_out_maxs = mac_out.clone()
        mac_out_maxs[avail_actions == 0] = -9999999

        target_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
        max_actions_qvals, max_actions_current = mac_out_maxs[:, :].max(dim=3,
                                                                        keepdim=True)

        if self.args.mixer == "qtran_base":
            joint_qs, vs = self.mixer(batch[:, :-1], mac_hidden_states[:,:-1])

            if self.args.double_q:
                max_actions_current_ = th.zeros(size=(batch.batch_size,
                                                      batch.max_seq_length,
                                                      self.args.n_agents,
                                                      self.args.n_actions),
                                                      device=batch.device)
                max_actions_current_onehot = max_actions_current_.scatter(3,
                                                                          max_actions_current[:, :],
                                                                          1)
                max_actions_onehot = max_actions_current_onehot
            else:
                max_actions = th.zeros(size=(batch.batch_size,
                                             batch.max_seq_length,
                                             self.args.n_agents,
                                             self.args.n_actions),
                                             device=batch.device)
                max_actions_onehot = max_actions.scatter(3,
                                                         target_max_actions[:, :],
                                                         1)
            target_joint_qs, target_vs = self.target_mixer(batch[:, 1:],
                                                           hidden_states=target_mac_hidden_states[:,1:],
                                                           actions=max_actions_onehot[:,1:])

            td_targets = rewards.reshape(-1,1)
            + self.args.gamma * (1
                                 - terminated.reshape(-1, 1)) * target_joint_qs
            td_error = (joint_qs - td_targets.detach())
            masked_td_error = td_error * mask.reshape(-1, 1)
            td_loss = (masked_td_error ** 2).sum() / mask.sum()
            if not self.args.double_q:
                max_actions_current_ = th.zeros(size=(batch.batch_size,
                                                      batch.max_seq_length,
                                                      self.args.n_agents,
                                                      self.args.n_actions),
                                                      device=batch.device )
                max_actions_current_onehot = max_actions_current_.scatter(3,
                                                                          max_actions_current[:, :],
                                                                          1)
            max_joint_qs, _ = self.mixer(batch[:, :-1],
                                         mac_hidden_states[:,:-1],
                                         actions=max_actions_current_onehot[:,:-1])

            opt_error = max_actions_qvals[:,:-1].sum(dim=2).reshape(-1, 1)
            - max_joint_qs.detach() + vs
            masked_opt_error = opt_error * mask.reshape(-1, 1)
            opt_loss = (masked_opt_error ** 2).sum() / mask.sum()
            nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1)
            - joint_qs.detach() + vs
            nopt_error = nopt_values.clamp(max=0)
            masked_nopt_error = nopt_error * mask.reshape(-1, 1)
            nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()
  
        elif self.args.mixer == "qtran_alt":
            raise Exception("Not supported yet.")

        loss = td_loss + self.args.opt_loss * opt_loss
        + self.args.nopt_min_loss * nopt_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params,
                                                self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num
            - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("opt_loss", opt_loss.item(), t_env)
            self.logger.log_stat("nopt_loss", nopt_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if self.args.mixer == "qtran_base":
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs",
                                     (masked_td_error.abs().sum().item()
                                      /mask_elems), t_env)
                self.logger.log_stat("td_targets",
                                     ((masked_td_error).sum().item()
                                      /mask_elems), t_env)
                self.logger.log_stat("td_chosen_qs",
                                     (joint_qs.sum().item()
                                      /mask_elems), t_env)
                self.logger.log_stat("v_mean",
                                     (vs.sum().item()
                                      /mask_elems), t_env)
                self.logger.log_stat("agent_indiv_qs",
                                     ((chosen_action_qvals * mask).sum().item()
                                      /(mask_elems * self.args.n_agents)), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        """Updates the target network."""
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        """Moves tensor to cuda."""
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

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
