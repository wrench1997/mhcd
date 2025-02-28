import numpy as np
import pygame
import time
import gymnasium as gym
from gymnasium import spaces

class MazeEnv(gym.Env):
    def __init__(self, level=1, partial_observe=True, render_mode=None):
        self.level = level
        self.partial_observe = partial_observe
        self.render_mode = render_mode
        self.predefined_mazes = self._create_predefined_mazes()
        self.cell_size = 30  # 每个格子的大小
        self.speed = 2.0     # 正常移动速度（像素/帧）
        self.run_speed = 4.0 # 加速时的速度
        self.double_click_time = 0.3  # 双击检测时间窗口（秒）
        
        # RL环境相关设置
        self.maze = self.predefined_mazes[self.level].copy()
        self.size = self.maze.shape[0]
        self.view_range = 2 if self.partial_observe else self.size
        
        # 定义观察空间
        obs_shape = (2*self.view_range+1, 2*self.view_range+1)
        self.observation_space = spaces.Dict({
            'map': spaces.Box(low=-1, high=1, shape=obs_shape, dtype=np.float32),
            'fps': spaces.Box(low=0, high=120, shape=(1,), dtype=np.float32)
        })
        
        # 定义动作空间: [up, down]的连续值 (-1.0 到 1.0)
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.action_space = spaces.Discrete(5)        
        # Pygame相关
        if self.render_mode == 'human':
            pygame.init()
            self.screen_width = 600
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("RL Maze")
            self.clock = pygame.time.Clock()
        
        self.reset()

    def _create_predefined_mazes(self):
        # 定义预设迷宫
        mazes = {}
        maze1 = np.zeros((8, 8))
        obstacles1 = [(1, 2), (1, 5), (2, 1), (2, 3), (2, 6), (3, 3), (3, 5),
                      (4, 1), (4, 3), (4, 6), (5, 3), (5, 5), (6, 2), (6, 4)]
        for x, y in obstacles1:
            maze1[x, y] = 1
        mazes[1] = maze1
        return mazes

    def reset(self):
        self.maze = self.predefined_mazes[self.level].copy()
        self.size = self.maze.shape[0]
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        self.current_pos = [self.start_pos[0] * self.cell_size, self.start_pos[1] * self.cell_size]
        self.velocity = [0, 0]
        self.key_states = {'left': False, 'right': False, 'up': False, 'down': False}
        self.key_press_times = {'left': [], 'right': [], 'up': [], 'down': []}
        self.is_running = False
        self.view_range = 2 if self.partial_observe else self.size
        self.steps = 0
        self.current_fps = 60  # 初始FPS
        
        # For RL
        observation = self._get_observation()
        info = {}
        
        if self.render_mode == 'human':
            self._render_frame()
            
        return observation, info

    def _get_observation(self):
        x, y = int(self.current_pos[0] // self.cell_size), int(self.current_pos[1] // self.cell_size)
        obs = np.ones((2*self.view_range+1, 2*self.view_range+1)) * -1
        for i in range(-self.view_range, self.view_range+1):
            for j in range(-self.view_range, self.view_range+1):
                ni, nj = x + i, y + j
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    obs[i+self.view_range, j+self.view_range] = self.maze[ni, nj]
        
        # 添加目标位置的相对位置信息
        # 对于RL环境，返回字典格式的观察
        return {
            'map': obs.astype(np.float32),
            'fps': np.array([self.current_fps], dtype=np.float32)
        }
    
    def step(self, action=None):
        # 如果传入 action（强化学习控制）
        if action is not None:
            # 重置所有按键状态
            for key in self.key_states:
                self.key_states[key] = False
                
            # 根据离散动作设置对应的按键状态
            if action == 1:  # 左
                self.key_states['left'] = True
            elif action == 2:  # 右
                self.key_states['right'] = True
            elif action == 3:  # 上
                self.key_states['up'] = True
            elif action == 4:  # 下
                self.key_states['down'] = True
                
        # 根据当前按键状态更新速度
        current_speed = self.run_speed if self.is_running else self.speed
        self.velocity = [0, 0]  # 重置速度
        
        if self.key_states['left']:
            self.velocity[1] = -current_speed
        if self.key_states['right']:
            self.velocity[1] = current_speed
        if self.key_states['up']:
            self.velocity[0] = -current_speed
        if self.key_states['down']:
            self.velocity[0] = current_speed

        # ...其余代码保持不变...
        new_pos = [self.current_pos[0] + self.velocity[0], self.current_pos[1] + self.velocity[1]]
        new_grid_x = int(new_pos[0] // self.cell_size)
        new_grid_y = int(new_pos[1] // self.cell_size)
        if (0 <= new_grid_x < self.size and 0 <= new_grid_y < self.size and 
            self.maze[new_grid_x, new_grid_y] == 0):
            self.current_pos = new_pos
        else:
            self.velocity = [0, 0]
            reward = -1.0  # 撞墙惩罚

        grid_pos = (int(self.current_pos[0] // self.cell_size), int(self.current_pos[1] // self.cell_size))
        done = grid_pos == self.goal_pos
        
        if done:
            reward = 10.0  # 到达目标奖励
        elif not 'reward' in locals():  # 如果没有撞墙
            reward = -0.1  # 每步小惩罚

        return self._get_observation(), reward, done, False, {}

    def _process_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
        return False

    def render(self):
        if self.render_mode == 'human':
            return self._render_frame()

    def _render_frame(self):
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen_width = 600
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("RL Maze")
            self.clock = pygame.time.Clock()
            
        # 获取屏幕尺寸
        screen_width, screen_height = self.screen.get_size()
        maze_width = self.size * self.cell_size
        maze_height = self.size * self.cell_size

        # 计算相机位置，以角色为中心
        camera_x = self.current_pos[1] - screen_width / 2
        camera_y = self.current_pos[0] - screen_height / 2

        # 限制相机位置在迷宫范围内
        if maze_width > screen_width:
            camera_x = max(0, min(camera_x, maze_width - screen_width))
        else:
            camera_x = 0
        if maze_height > screen_height:
            camera_y = max(0, min(camera_y, maze_height - screen_height))
        else:
            camera_y = 0

        # 清空屏幕
        self.screen.fill((255, 255, 255))

        # 渲染迷宫
        for i in range(self.size):
            for j in range(self.size):
                screen_x = j * self.cell_size - camera_x
                screen_y = i * self.cell_size - camera_y
                # 只渲染屏幕可见区域的格子
                if -self.cell_size <= screen_x < screen_width and -self.cell_size <= screen_y < screen_height:
                    if self.maze[i, j] == 1:
                        pygame.draw.rect(self.screen, (0, 0, 0), 
                                      (screen_x, screen_y, self.cell_size, self.cell_size))
                    pygame.draw.rect(self.screen, (200, 200, 200), 
                                   (screen_x, screen_y, self.cell_size, self.cell_size), 1)

        # 绘制起点和终点
        start_screen_x = self.start_pos[1] * self.cell_size - camera_x
        start_screen_y = self.start_pos[0] * self.cell_size - camera_y
        goal_screen_x = self.goal_pos[1] * self.cell_size - camera_x
        goal_screen_y = self.goal_pos[0] * self.cell_size - camera_y

        if -self.cell_size <= start_screen_x < screen_width and -self.cell_size <= start_screen_y < screen_height:
            pygame.draw.rect(self.screen, (0, 255, 0), 
                          (start_screen_x, start_screen_y, self.cell_size, self.cell_size))
        if -self.cell_size <= goal_screen_x < screen_width and -self.cell_size <= goal_screen_y < screen_height:
            pygame.draw.rect(self.screen, (255, 0, 0), 
                          (goal_screen_x, goal_screen_y, self.cell_size, self.cell_size))

        # 绘制角色
        player_screen_x = self.current_pos[1] - camera_x
        player_screen_y = self.current_pos[0] - camera_y
        pygame.draw.circle(self.screen, (0, 0, 255), 
                         (int(player_screen_x), int(player_screen_y)), self.cell_size // 3)

        # 部分可观察迷雾
        if self.partial_observe:
            fog = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            fog.fill((0, 0, 0, 100))  # 调低透明度 (原来是180)
            x, y = int(self.current_pos[0] // self.cell_size), int(self.current_pos[1] // self.cell_size)
            for i in range(-self.view_range, self.view_range + 1):
                for j in range(-self.view_range, self.view_range + 1):
                    ni, nj = x + i, y + j
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        screen_x = nj * self.cell_size - camera_x
                        screen_y = ni * self.cell_size - camera_y
                        if -self.cell_size <= screen_x < screen_width and -self.cell_size <= screen_y < screen_height:
                            pygame.draw.rect(fog, (0, 0, 0, 0), 
                                             (screen_x, screen_y, self.cell_size, self.cell_size))
            self.screen.blit(fog, (0, 0))
            
        # 显示FPS
        font = pygame.font.Font(None, 36)
        fps_text = font.render(f"FPS: {int(self.current_fps)}", True, (0, 0, 0))
        self.screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)
        return self.screen

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()


    def update_keys(self, events):
        current_time = time.time()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.key_states['left'] = True
                    self.key_press_times['left'].append(current_time)
                elif event.key == pygame.K_RIGHT:
                    self.key_states['right'] = True
                    self.key_press_times['right'].append(current_time)
                elif event.key == pygame.K_UP:
                    self.key_states['up'] = True
                    self.key_press_times['up'].append(current_time)
                elif event.key == pygame.K_DOWN:
                    self.key_states['down'] = True
                    self.key_press_times['down'].append(current_time)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    self.key_states['left'] = False
                elif event.key == pygame.K_RIGHT:
                    self.key_states['right'] = False
                elif event.key == pygame.K_UP:
                    self.key_states['up'] = False
                elif event.key == pygame.K_DOWN:
                    self.key_states['down'] = False

        for direction in ['left', 'right', 'up', 'down']:
            presses = self.key_press_times[direction]
            while presses and current_time - presses[0] > self.double_click_time:
                presses.pop(0)
            if len(presses) >= 2 and self.key_states[direction]:
                self.is_running = True
            if not any(self.key_states.values()):
                self.is_running = False

