import numpy as np
import pygame
import time

class MazeEnv:
    def __init__(self, level=1, partial_observe=True):
        self.level = level
        self.partial_observe = partial_observe
        self.predefined_mazes = self._create_predefined_mazes()
        self.cell_size = 30  # 每个格子的大小
        self.speed = 2.0     # 正常移动速度（像素/帧）
        self.run_speed = 4.0 # 加速时的速度
        self.double_click_time = 0.3  # 双击检测时间窗口（秒）
        self.reset()

    def _create_predefined_mazes(self):
        # 与原代码相同，定义预设迷宫
        mazes = {}
        maze1 = np.zeros((8, 8))
        obstacles1 = [(1, 2), (1, 5), (2, 1), (2, 3), (2, 6), (3, 3), (3, 5),
                      (4, 1), (4, 3), (4, 6), (5, 3), (5, 5), (6, 2), (6, 4)]
        for x, y in obstacles1:
            maze1[x, y] = 1
        mazes[1] = maze1
        # 其他关卡（maze2, maze3）省略，与原代码相同
        return mazes

    def reset(self):
        self.maze = self.predefined_mazes[self.level].copy()
        self.size = self.maze.shape[0]
        self.start_pos = (0, 0)
        self.goal_pos = (self.size-1, self.size-1)
        
        # 角色位置使用浮点数表示（像素坐标）
        self.current_pos = [self.start_pos[0] * self.cell_size, self.start_pos[1] * self.cell_size]
        self.velocity = [0, 0]  # 当前速度 [vx, vy]
        
        # 按键状态和双击检测
        self.key_states = {'left': False, 'right': False, 'up': False, 'down': False}
        self.key_press_times = {'left': [], 'right': [], 'up': [], 'down': []}  # 记录按键时间
        self.is_running = False  # 是否处于加速状态
        
        self.view_range = 2 if self.partial_observe else self.size
        return self._get_observation()

    def _get_observation(self):
        # 计算角色所在的网格坐标
        x, y = int(self.current_pos[0] // self.cell_size), int(self.current_pos[1] // self.cell_size)
        obs = np.ones((2*self.view_range+1, 2*self.view_range+1)) * -1
        for i in range(-self.view_range, self.view_range+1):
            for j in range(-self.view_range, self.view_range+1):
                ni, nj = x + i, y + j
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    obs[i+self.view_range, j+self.view_range] = self.maze[ni, nj]
        return obs

    def step(self):
        # 更新速度基于按键状态
        self.velocity = [0, 0]
        current_speed = self.run_speed if self.is_running else self.speed
        
        if self.key_states['left']:
            self.velocity[1] = -current_speed
        if self.key_states['right']:
            self.velocity[1] = current_speed
        if self.key_states['up']:
            self.velocity[0] = -current_speed
        if self.key_states['down']:
            self.velocity[0] = current_speed

        # 更新位置
        new_pos = [self.current_pos[0] + self.velocity[0], self.current_pos[1] + self.velocity[1]]
        
        # 碰撞检测
        new_grid_x, new_grid_y = int(new_pos[0] // self.cell_size), int(new_pos[1] // self.cell_size)
        if (0 <= new_grid_x < self.size and 0 <= new_grid_y < self.size and 
            self.maze[new_grid_x, new_grid_y] == 0):
            self.current_pos = new_pos
        else:
            self.velocity = [0, 0]  # 碰到障碍物或边界停止

        # 检查是否到达目标
        grid_pos = (int(self.current_pos[0] // self.cell_size), int(self.current_pos[1] // self.cell_size))
        done = grid_pos == self.goal_pos
        reward = 10 if done else -0.1
        
        return self._get_observation(), reward, done

    def update_keys(self, events):
        # 处理按键事件
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

        # 检测双击并更新加速状态
        for direction in ['left', 'right', 'up', 'down']:
            presses = self.key_press_times[direction]
            # 清理过期时间戳
            while presses and current_time - presses[0] > self.double_click_time:
                presses.pop(0)
            # 检查是否双击
            if len(presses) >= 2 and self.key_states[direction]:
                self.is_running = True
            # 如果没有按键按下，取消加速
            if not any(self.key_states.values()):
                self.is_running = False

    def render(self, surface):
        surface.fill((255, 255, 255))
        for i in range(self.size):
            for j in range(self.size):
                if self.maze[i, j] == 1:
                    pygame.draw.rect(surface, (0, 0, 0), 
                                     (j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size))
                pygame.draw.rect(surface, (200, 200, 200), 
                                 (j*self.cell_size, i*self.cell_size, self.cell_size, self.cell_size), 1)
        
        # 绘制起点和终点
        pygame.draw.rect(surface, (0, 255, 0), 
                         (self.start_pos[1]*self.cell_size, self.start_pos[0]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(surface, (255, 0, 0), 
                         (self.goal_pos[1]*self.cell_size, self.goal_pos[0]*self.cell_size, self.cell_size, self.cell_size))
        
        # 绘制角色（平滑位置）
        pygame.draw.circle(surface, (0, 0, 255), 
                           (int(self.current_pos[1]), int(self.current_pos[0])), self.cell_size//3)
        
        # 部分可观察迷雾
        if self.partial_observe:
            fog = pygame.Surface((self.size*self.cell_size, self.size*self.cell_size), pygame.SRCALPHA)
            fog.fill((0, 0, 0, 180))
            x, y = int(self.current_pos[0] // self.cell_size), int(self.current_pos[1] // self.cell_size)
            for i in range(-self.view_range, self.view_range+1):
                for j in range(-self.view_range, self.view_range+1):
                    ni, nj = x + i, y + j
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        pygame.draw.rect(fog, (0, 0, 0, 0), 
                                         (nj*self.cell_size, ni*self.cell_size, self.cell_size, self.cell_size))
            surface.blit(fog, (0, 0))

# 测试代码
def play_game():
    pygame.init()
    env = MazeEnv(level=1)
    screen = pygame.display.set_mode((env.size * env.cell_size, env.size * env.cell_size))
    pygame.display.set_caption("DFO风格迷宫")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
        
        env.update_keys(events)
        obs, reward, done = env.step()
        env.render(screen)
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()

if __name__ == "__main__":
    play_game()