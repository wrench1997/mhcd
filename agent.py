import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pygame



def train_ddpg(env_fn, state_processor_fn, 
              episodes=1000, max_steps=500, batch_size=256, 
              render_interval=100, save_interval=500, eval_interval=100):
    
    # Create environment
    env = env_fn()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get state and action dimensions
    state = env.reset()
    obs_sample = state_processor_fn(state)
    state_dim = obs_sample.shape[0]  # Flattened state dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize DDPG agent
    agent = DDPG(state_dim, action_dim, max_action, device)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6), device=device)
    
    # Create directory for saving models and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./results/maze_ddpg_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Track training progress
    episode_rewards = []
    episode_steps = []
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        
        # Render first episode and then every render_interval episodes
        render = episode == 1 or episode % render_interval == 0
        
        while not done and episode_step < max_steps:
            episode_step += 1
            
            # Process state for input to agent
            processed_state = state_processor_fn(state)
            
            # Select action with noise for exploration
            action = agent.select_action(processed_state, noise=0.1)
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Store transition in replay buffer
            replay_buffer.add(state, action, next_state, reward, done)
            
            # Update the agent
            if replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Render if needed
            if render:
                env.render()
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        # Print progress
        print(f"Episode: {episode}/{episodes}, Reward: {episode_reward:.2f}, Steps: {episode_step}")
        
        # Save model periodically
        if episode % save_interval == 0:
            agent.save(f"{save_dir}/model_{episode}")
            
        # Evaluate the agent periodically
        if episode % eval_interval == 0:
            eval_reward = evaluate_agent(agent, env_fn, state_processor_fn, episodes=5)
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

def evaluate_agent(agent, env_fn, state_processor_fn, episodes=10, render=False):
    env = env_fn()
    total_rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            processed_state = state_processor_fn(state)
            
            # Select action without noise
            action = agent.select_action(processed_state, noise=0.0)
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Render if needed
            if render:
                env.render()
        
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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get state and action dimensions
    state = env.reset()
    obs_sample = state_processor_fn(state)
    state_dim = obs_sample.shape[0]  # Flattened state dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize DDPG agent
    agent = DDPG(state_dim, action_dim, max_action, device)
    
    # Load trained model
    agent.load(model_path)
    
    # Track test results
    episode_rewards = []
    episode_steps = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False
        
        while not done:
            episode_step += 1
            
            # Process state for input to agent
            processed_state = state_processor_fn(state)
            
            # Select action without noise (deterministic policy)
            action = agent.select_action(processed_state, noise=0.0)
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Render if needed
            if render:
                env.render()
                
            # Optional delay for visualization
            if render:
                import time
                time.sleep(0.01)
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        # Print results
        print(f"Episode: {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {episode_step}")
    
    env.close()
    return episode_rewards, episode_steps


