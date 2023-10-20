import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupMixer(nn.Module):
    def __init__(self, args):
        super(SupMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.surp_dim = 1

        self.embed_dim = 4
        hypernet_embed = 256
        self.state_net = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        self.q_net = nn.Sequential(nn.Linear(1, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        self.surp_net = nn.Sequential(nn.Linear(self.surp_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))

        self.main_net = nn.Sequential(nn.Linear(self.embed_dim*3, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, 1))


    def forward(self, agent_qs, states, surp):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1)
        surp = surp.view(-1, 1)
        # Apply Networks
        state_out = self.state_net(states)
        q_out = self.q_net(agent_qs)
        surp_out = self.surp_net(surp)
        main_ip = th.cat([state_out,surp_out,q_out], dim=1)
        q_surp = self.main_net(main_ip).view(bs, -1)
        return q_surp


