import pygame
from DFO import MazeEnv


def play_game():
    # Initialize pygame
    pygame.init()
    
    # Create environment
    env = MazeEnv(level=1, partial_observe=True, render_mode='human')
    
    # Set up display
    screen_width = 600
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Maze Game (Human Play)")
    clock = pygame.time.Clock()
    
    # Game variables
    running = True
    game_over = False
    state = env.reset()
    
    # Track for display
    total_reward = 0
    steps = 0
    
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            
            # Handle key presses for human play
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    env.key_states['left'] = True
                elif event.key == pygame.K_RIGHT:
                    env.key_states['right'] = True
                elif event.key == pygame.K_UP:
                    env.key_states['up'] = True
                elif event.key == pygame.K_DOWN:
                    env.key_states['down'] = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    env.key_states['left'] = False
                elif event.key == pygame.K_RIGHT:
                    env.key_states['right'] = False
                elif event.key == pygame.K_UP:
                    env.key_states['up'] = False
                elif event.key == pygame.K_DOWN:
                    env.key_states['down'] = False
        
        if not game_over:
            # For human play, convert key states to compatible action format
            # (We'll just use manual controls, not the step() method with actions)
            action = [0, 0]  # Placeholder, not actually used

            action = [0.0, 0.5]  # 默认向右移动
            if env.key_states['up']:
                action[0] -= 1.0  # 向上移动
            if env.key_states['down']:
                action[0] += 1.0  # 向下移动

            # Update environment based on key states
            # env.update_keys(events)
            observation, reward, terminated, truncated, info = env.step()
            env.render()

            done = terminated or truncated

            
            # Update tracking variables
            total_reward += reward
            steps += 1
            
            # Display stats
            font = pygame.font.Font(None, 36)
            reward_text = font.render(f"Reward: {total_reward:.2f}", True, (0, 0, 0))
            steps_text = font.render(f"Steps: {steps}", True, (0, 0, 0))
            # screen.blit(reward_text, (10, 50))
            # screen.blit(steps_text, (10, 90))
            
            pygame.display.flip()
            
            if done:
                font = pygame.font.Font(None, 50)
                text = font.render("Success!", True, (255, 0, 0))
                # text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
                # screen.blit(text, text_rect)
                pygame.display.flip()
                game_over = True
                pygame.time.delay(3000)  # Display for 3 seconds
                running = False
        
        clock.tick(60)  # 60 FPS
    
    pygame.quit()

