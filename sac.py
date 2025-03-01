import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy  as np
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
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 her_k=4):  # 每个经验生成k个HER样本
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        self.her_k = her_k
        self.episode_trajectory = []  # 添加轨迹缓存
        
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
    
    def train(self, replay_buffer, batch_size=256, beta=0.4):
        # 从优先级buffer采样
        state_batch, action_batch, reward_batch, next_state_batch, \
        mask_batch, indices, weights = replay_buffer.sample(batch_size, beta)
        
        with torch.no_grad():
            next_state_value = self._get_value(next_state_batch)
            target_q = reward_batch + mask_batch * self.gamma * next_state_value
            
        # Critic loss with importance sampling weights
        current_q1, current_q2 = self.critic(state_batch)
        
        # 确保所有张量维度正确
        current_q1 = current_q1.view(batch_size, -1)  # [batch_size, 1]
        current_q2 = current_q2.view(batch_size, -1)  # [batch_size, 1]
        target_q = target_q.view(batch_size, -1)      # [batch_size, 1]
        
        # 计算TD误差，确保得到[batch_size]形状
        td_error1 = torch.abs(current_q1 - target_q).mean(dim=1)  # [batch_size]
        td_error2 = torch.abs(current_q2 - target_q).mean(dim=1)  # [batch_size]
        
        # 检查维度
        assert td_error1.shape == (batch_size,), f"TD error1 shape: {td_error1.shape}"
        assert td_error2.shape == (batch_size,), f"TD error2 shape: {td_error2.shape}"
        
        critic_loss = (weights * (td_error1 + td_error2)).mean()
        
        # 更新优先级 - 使用最大的TD误差
        priorities = torch.max(td_error1, td_error2).detach()
        
        # 再次确认priorities维度
        assert priorities.shape == (batch_size,), f"Priorities shape: {priorities.shape}"
        replay_buffer.update_priorities(indices, priorities)
        
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
        # 保存模型和优化器状态
        torch.save({
            'episode': self.current_episode,
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optim.state_dict()
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.current_episode = checkpoint['episode']
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optim.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        # 重新初始化目标网络
        self.critic_target = Critic(self.critic.q1[0].in_features).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def store_experience(self, replay_buffer, state, action, next_state, reward, done, 
                        goal, achieved_goal):
        # 存储原始经验和轨迹
        replay_buffer.add(state, action, next_state, reward, done, goal, achieved_goal)
        
        # 存储当前transition到轨迹中
        transition = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'achieved_goal': achieved_goal,
            'reward': reward,
            'done': done
        }
        self.episode_trajectory.append(transition)
        
        # 对最近k步应用HER
        if len(self.episode_trajectory) > self.her_k:
            recent_transitions = self.episode_trajectory[-self.her_k:]
            # 使用最后一个状态作为虚拟目标
            virtual_goal = recent_transitions[-1]['achieved_goal']
            
            # 对前面k-1步重新计算奖励并存储
            for t in range(len(recent_transitions) - 1):
                curr_transition = recent_transitions[t]
                # 使用虚拟目标重新计算奖励
                new_reward = self.compute_reward(
                    curr_transition['achieved_goal'],
                    virtual_goal
                )
                # 存储新的经验
                replay_buffer.add(
                    curr_transition['state'],
                    curr_transition['action'],
                    curr_transition['next_state'],
                    new_reward,
                    t == len(recent_transitions) - 2,  # 倒数第二步设为done
                    virtual_goal,
                    curr_transition['achieved_goal']
                )
        
        # 修改HER策略,随机采样future goals
        if len(self.episode_trajectory) > self.her_k:
            recent_transitions = self.episode_trajectory[-self.her_k:]
            # 随机选择k个未来时刻作为目标
            future_indices = np.random.choice(
                range(len(recent_transitions)), 
                size=min(self.her_k, len(recent_transitions)),
                replace=False
            )
            
            for future_idx in future_indices:
                virtual_goal = recent_transitions[future_idx]['achieved_goal']
                # 只对future_idx之前的步骤应用HER
                for t in range(future_idx):
                    curr_transition = recent_transitions[t]
                    new_reward = self.compute_reward(
                        curr_transition['achieved_goal'],
                        virtual_goal
                    )
                    replay_buffer.add(
                        curr_transition['state'],
                        curr_transition['action'],
                        curr_transition['next_state'],
                        new_reward,
                        t == future_idx - 1,
                        virtual_goal,
                        curr_transition['achieved_goal']
                    )
        
        if done:
            # 清空轨迹缓存
            self.episode_trajectory.clear()
                
    def compute_reward(self, achieved_goal, desired_goal):
        achieved_goal = np.array(achieved_goal)
        desired_goal = np.array(desired_goal)
        # 根据实际情况实现奖励计算
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -1.0 if distance > 0.05 else 0.0

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

class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e7), device="cpu", alpha=0.6, beta=0.4):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.epsilon = 1e-6 # 防止优先级为0
        
        if isinstance(state_dim, dict):
            self.map_shape = state_dim['map']
            flat_state_dim = np.prod(self.map_shape) + state_dim['fps'][0] + state_dim['time'][0]
        else:
            flat_state_dim = state_dim
            
        # 存储结构
        self.state = torch.zeros((max_size, flat_state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros(max_size, dtype=torch.long, device=device)
        self.next_state = torch.zeros((max_size, flat_state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.not_done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        
        # HER相关
        self.goal = torch.zeros((max_size, 2), dtype=torch.float32, device=device)
        self.achieved_goal = torch.zeros((max_size, 2), dtype=torch.float32, device=device)
        
        # 优先级存储
        self.priorities = np.zeros(max_size)

    def add(self, state, action, next_state, reward, done, goal=None, achieved_goal=None):
        if isinstance(state, dict):
            flat_state = np.concatenate([state['map'].flatten(), state['fps'] + state['time']])
            flat_next_state = np.concatenate([next_state['map'].flatten(), next_state['fps'] + state['time']])
        else:
            flat_state = state
            flat_next_state = next_state
            
        self.state[self.ptr] = torch.FloatTensor(flat_state)
        self.action[self.ptr] = torch.tensor(action, dtype=torch.long)
        self.next_state[self.ptr] = torch.FloatTensor(flat_next_state)
        self.reward[self.ptr] = torch.FloatTensor([reward])
        self.not_done[self.ptr] = torch.FloatTensor([1.0 - float(done)])
        
        if goal is not None:
            # 确保goal是2维坐标
            if isinstance(goal, tuple):
                goal = np.array(goal)
            self.goal[self.ptr] = torch.FloatTensor(goal)
        if achieved_goal is not None:
            # 确保achieved_goal是2维坐标
            if isinstance(achieved_goal, tuple):
                achieved_goal = np.array(achieved_goal)
            self.achieved_goal[self.ptr] = torch.FloatTensor(achieved_goal)
            
        # 修改优先级计算,同时考虑reward和之前的优先级
        reward_priority = abs(reward)  # 奖励的绝对值作为优先级
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        # 将reward优先级和历史优先级结合
        combined_priority = 0.5 * reward_priority + 0.5 * max_priority
        self.priorities[self.ptr] = combined_priority + self.epsilon
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta
            
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 重要性采样
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)
        
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.not_done[indices],
            indices,
            weights
        )

    def update_priorities(self, indices, td_errors):
        # 结合TD误差和存储的reward来更新优先级
        td_priorities = np.abs(td_errors.cpu().numpy())
        reward_priorities = np.abs(self.reward[indices].cpu().numpy())
        # 组合两种优先级
        combined_priorities = 0.5 * td_priorities + 0.5 * reward_priorities.flatten()
        self.priorities[indices] = combined_priorities + self.epsilon
