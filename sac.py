import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        logits = self.net(state)
        return logits
    
    def sample(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        # Compute log probability
        log_prob = dist.log_prob(action)
        
        return action, log_prob, probs

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.q1(state), self.q2(state)

class DiscreteSAC:
    def __init__(self, state_dim, action_dim, device,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.critic_target = Critic(state_dim).to(device)
        
        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Automatic entropy tuning
        self.target_entropy = -0.98 * np.log(1.0 / action_dim) 
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if evaluate:
                logits = self.actor(state)
                action = torch.argmax(logits, dim=-1)
            else:
                action, _, _ = self.actor.sample(state)
            return action.cpu().data.numpy().item()
    
    def train(self, replay_buffer, batch_size=256):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            next_state_value = self._get_value(next_state_batch)
            target_q = reward_batch + mask_batch * self.gamma * next_state_value
            
        # Critic loss
        current_q1, current_q2 = self.critic(state_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor loss
        pi, log_pi, _ = self.actor.sample(state_batch)
        actor_q1, actor_q2 = self.critic(state_batch)
        min_q = torch.min(actor_q1, actor_q2)
        
        actor_loss = (self.alpha * log_pi - min_q).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        # Optimize alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        
        # Soft update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def _get_value(self, state):
        with torch.no_grad():
            action_probs = F.softmax(self.actor(state), dim=-1)
            next_action, next_log_pi, _ = self.actor.sample(state) 
            target_q1, target_q2 = self.critic_target(state)
            min_q = torch.min(target_q1, target_q2)
            return min_q - self.alpha * next_log_pi
            
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor")) 
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.critic_target = Critic(self.critic.q1[0].in_features).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e7), device="cpu"):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        if isinstance(state_dim, dict):
            self.map_shape = state_dim['map']
            flat_state_dim = np.prod(self.map_shape) + state_dim['fps'][0] + state_dim['time'][0]
        else:
            flat_state_dim = state_dim
            
        self.state = torch.zeros((max_size, flat_state_dim), dtype=torch.float32, device=device)
        # 修改action为长整型，存储离散动作索引
        self.action = torch.zeros(max_size, dtype=torch.long, device=device)
        self.next_state = torch.zeros((max_size, flat_state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.not_done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
    
    def add(self, state, action, next_state, reward, done):
        if isinstance(state, dict):
            flat_state = np.concatenate([state['map'].flatten(), state['fps'] + state['time']])
            flat_next_state = np.concatenate([next_state['map'].flatten(), next_state['fps'] + state['time']])
        else:
            flat_state = state
            flat_next_state = next_state
            
        self.state[self.ptr] = torch.FloatTensor(flat_state)
        # 直接存储动作索引
        self.action[self.ptr] = torch.tensor(action, dtype=torch.long)
        self.next_state[self.ptr] = torch.FloatTensor(flat_next_state)
        self.reward[self.ptr] = torch.FloatTensor([reward])
        self.not_done[self.ptr] = torch.FloatTensor([1.0 - float(done)])
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.not_done[ind]
        )
