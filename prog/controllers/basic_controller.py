"""Implements the basic controller for multi-agent learning."""

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class BasicMAC:
    """Basic controller with shared parameters.
    
    Attributes:
        n_agents: number of agents.
        args: experiment arguments.
        agent_output_type: action type of agent.
        action_selector: action selection strategy.
        hidden_states: hidden states from past timesteps.
    """

    def __init__(self, scheme, groups, args):
        """Initializes the controller object."""
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env,
                       bs=slice(None), test_mode=False):
        """Select actions from the controller.
        
        Args:
            ep_batch: batch of episodes.
            t_ep: timestep of episode.
            t_env: timstep of env.
            bs: batch size.
            test_mode: whether testing or not.
        """
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs],
                                                            avail_actions[bs],
                                                            t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        """Step through the controller.
        
        Args:
            ep_batch: batch of episodes.
            t: timstep.
            test_mode: whether testing or not.
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs,
                                                    self.hidden_states)

        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1,
                                                                    keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon
                               /epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        """Initialize hidden state.
        
        Args:
            batch_size: size of a batch of samples.
        """
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1)

    def parameters(self):
        """Returns parameters of the agent."""
        return self.agent.parameters()

    def load_state(self, other_mac):
        """Loads current parameter state.
        
        Args:
        other_mac: different multi-agent controller.
        """
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        """Move agent to cuda device."""
        self.agent.cuda()

    def save_models(self, path):
        """Save models.
        
        Args:
            path: logging directory.
        """
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """Load models.
        
        Args:
            path: logging directory.
        """
        self.agent.load_state_dict(th.load("{}/agent.th".format(path),
                                           map_location=lambda storage,
                                           loc: storage))

    def _build_agents(self, input_shape):
        """Builds the agents using registry calls.
        
        Args:
            input_shape: shape of the input observation.
        """
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """Builds the input observations.
        
        Args:
            batch: batch of data samples.
            t: timestep.
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents,
                                 device=batch.device).unsqueeze(0).expand(bs,
                                                                          -1,
                                                                          -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        """Returns the input shape.
        
        Args:
            scheme: buffer scheme.
        """
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
