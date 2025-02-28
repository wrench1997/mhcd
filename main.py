import argparse
from agent import train_ddpg, test_agent, process_state  # 从 agent 导入 process_state
from play import play_game
from DFO import MazeEnv  # 新增导入 MazeEnv

def env_factory(render_mode=None):
    return MazeEnv(level=1, partial_observe=True, render_mode=render_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Maze Game with DDPG')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'play'],
                       help='Mode: train, test, or play')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model (required for test mode)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes for training')
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train DDPG agent
        agent, rewards, steps = train_ddpg(
            env_fn=env_factory,
            state_processor_fn=process_state,
            episodes=args.episodes
        )
    elif args.mode == 'test':
        # Test trained agent
        if args.model_path is None:
            print("Error: model_path is required for test mode")
        else:
            rewards, steps = test_agent(
                model_path=args.model_path,
                env_fn=env_factory,
                state_processor_fn=process_state
            )
    elif args.mode == 'play':
        # Human gameplay
        play_game()
    else:
        print(f"Unknown mode: {args.mode}")
