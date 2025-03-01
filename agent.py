import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pygame
from sac import DiscreteSAC, PrioritizedReplayBuffer  # 修改导入语句



def train_sac(env_fn, state_processor_fn, 
              episodes=1000, max_steps=250, batch_size=256, 
              render_interval=100, save_interval=100, eval_interval=100,
              continue_training=False, load_path=None):
    
    # Create environment
    env = env_fn()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get state and action dimensions
    state,_ = env.reset()
    obs_sample = state_processor_fn(state)
    state_dim = obs_sample.shape[0]  # Flattened state dim
    action_dim = env.action_space.n
    
    # Initialize SAC agent with discrete actions
    agent = DiscreteSAC(state_dim, action_dim, device)
    
    # 如果继续训练，加载之前的模型
    start_episode = 1
    if continue_training and load_path:
        agent.load(load_path)
        start_episode = agent.current_episode + 1
        print(f"Continuing training from episode {start_episode}")
    
    # 将ReplayBuffer替换为PrioritizedReplayBuffer
    replay_buffer = PrioritizedReplayBuffer(
        state_dim=state_dim, 
        action_dim=action_dim, 
        max_size=int(1e7), 
        device=device,
        alpha=0.6,  # 优先级指数
        beta=0.4    # 初始重要性采样指数
    )
    
    # Create directory for saving models and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./results/maze_ddpg_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Track training progress
    episode_rewards = []
    episode_steps = []
    
    # 用于HER的轨迹缓存
    episode_trajectory = []
    
    for episode in range(start_episode, episodes + 1):
        state,_ = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        # 清空轨迹缓存
        episode_trajectory.clear()
        
        # Render first episode and then every render_interval episodes
        render = episode == 1 or episode % render_interval == 0
        
        while not done and episode_step < max_steps:
            episode_step += 1
            
            # Process state for input to agent
            processed_state = state_processor_fn(state)
            
            # Select action with noise for exploration
            action = agent.select_action(processed_state)
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # 调用store_experience存储经验
            agent.store_experience(
                replay_buffer,
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
                goal=env.goal_pos,           # 当前目标
                achieved_goal=env.current_pos # 当前位置
            )
            
            # Update the agent
            if replay_buffer.size > batch_size:
                # 计算当前beta值（从0.4线性增加到1.0）
                current_beta = min(1.0, 0.4 + episode / episodes * 0.6)
                agent.train(replay_buffer, batch_size, beta=current_beta)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Render if needed
            if render:
                env.render()
        
        # 在轨迹结束时应用HER
        if done and len(episode_trajectory) > 0:
            # 从轨迹中采样future goals
            future_indices = np.random.choice(
                len(episode_trajectory), 
                size=min(len(episode_trajectory), agent.her_k),
                replace=False
            )
            
            for future_idx in future_indices:
                future_state = episode_trajectory[future_idx]['state']
                achieved_goal = (
                    int(future_state['current_pos'][0] // env.cell_size),
                    int(future_state['current_pos'][1] // env.cell_size)
                )
                
                # 对轨迹中的每一步重新计算奖励并存储
                for step in episode_trajectory[:future_idx+1]:
                    current_achieved = (
                        int(step['state']['current_pos'][0] // env.cell_size),
                        int(step['state']['current_pos'][1] // env.cell_size)
                    )
                    # 使用新目标计算奖励
                    new_reward = 0.0 if current_achieved == achieved_goal else -1.0
                    
                    replay_buffer.add(
                        state=step['state'],
                        action=step['action'],
                        next_state=step['next_state'],
                        reward=new_reward,
                        done=step['done'],
                        goal=achieved_goal,
                        achieved_goal=current_achieved
                    )
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        # Print progress
        print(f"Episode: {episode}/{episodes}, Reward: {episode_reward:.2f}, Steps: {episode_step}")
        
        # 更新当前episode
        agent.current_episode = episode
        
        # Save model periodically
        if episode % save_interval == 0:
            save_path = f"{save_dir}/model_{episode}"
            agent.save(save_path)
            # 同时保存训练曲线
            np.save(f"{save_dir}/rewards_{episode}.npy", episode_rewards)
            np.save(f"{save_dir}/steps_{episode}.npy", episode_steps)
            
        # Evaluate the agent periodically
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env_fn, state_processor_fn, episodes=5,max_steps=max_steps)
            print(f"Evaluation Reward: {eval_reward:.2f}")
            
            # Plot training progress so far
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.subplot(1, 2, 2)
            plt.plot(episode_steps)
            plt.title('Episode Steps')
            plt.savefig(f"{save_dir}/training_progress_{episode}.png")
            plt.close()
    
    # Save the final model
    agent.save(f"{save_dir}/model_final")
    np.save(f"{save_dir}/rewards_final.npy", episode_rewards)
    np.save(f"{save_dir}/steps_final.npy", episode_steps)
    
    # Plot final training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.title('Episode Steps')
    plt.savefig(f"{save_dir}/training_progress_final.png")
    plt.close()
    
    env.close()
    return agent, episode_rewards, episode_steps

def evaluate_agent(agent, env_fn, state_processor_fn, episodes=10, render=False,max_steps=150):
    env = env_fn()
    total_rewards = []
    
    for _ in range(episodes):
        state,_ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            steps += 1
            processed_state = state_processor_fn(state)
            
            # Select action without noise
            action = agent.select_action(processed_state)
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Render if needed
            if render:
                env.render()
            if steps >= max_steps:
                done = True

        
        total_rewards.append(episode_reward)
    
    avg_reward = sum(total_rewards) / len(total_rewards)
    return avg_reward

# State processor function
def process_state(state):
    # Flatten and normalize the map part
    flat_map = state['map'].flatten() / 1.0  # Normalize to [-1,1]
    
    # Normalize the FPS part
    normalized_fps = state['fps'] / 120.0  # Normalize to [0,1]
    
    # Concatenate all parts
    processed_state = np.concatenate([flat_map, normalized_fps])
    return processed_state



def test_agent(model_path, env_fn, state_processor_fn, num_episodes=5, render=True):
    # Create environment
    env = env_fn(render_mode='human' if render else None)
    import time
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get state and action dimensions
    state,_ = env.reset()
    obs_sample = state_processor_fn(state)
    state_dim = obs_sample.shape[0]  # Flattened state dim
    action_dim = env.action_space.n
    # max_action = float(env.action_space.high[0])
    
    # Initialize DDPG agent
    agent = DiscreteSAC(state_dim, action_dim, device)
    
    # Load trained model
    agent.load(model_path)
    
    # Track test results
    episode_rewards = []
    episode_steps = []
    
    for episode in range(1, num_episodes + 1):
        state,_ = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        
        while not done:
            episode_step += 1
            
            # Process state for input to agent
            processed_state = state_processor_fn(state)

            for i in range(15):
                # Select action without noise (deterministic policy)
                action = agent.select_action(processed_state)
                # Execute action in environment
                next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            
            
            # Render if needed
            if render:
                time.sleep(0.01)
                env.render()
                
            # Optional delay for visualization
            #if render:
                # import time
                # time.sleep(0.01)
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        # Print results
        print(f"Episode: {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {episode_step}")
    
    env.close()
    return episode_rewards, episode_steps


