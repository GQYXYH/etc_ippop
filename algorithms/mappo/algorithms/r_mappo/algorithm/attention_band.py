import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
class ModifiedAttention(nn.Module):
    def __init__(self, obs_dim, num_agents):
        super(ModifiedAttention, self).__init__()
        self.query = nn.Linear(obs_dim, obs_dim)
        self.key = nn.Linear(obs_dim, obs_dim)
        self.value = nn.Linear(obs_dim, obs_dim)
        self.num_agents = num_agents

    def forward(self, observations):
        query = self.query(observations)
        key = self.key(observations)
        value = self.value(observations)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights