import pygame 
import random
import numpy as np
from enum import Enum
import gymnasium as gym 
from gymnasium import spaces
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

#colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
#env params
BLOCK_SIZE = 20
SPEED = 40

class SnakeEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': SPEED
    }


    def __init__(self, width=640, height=480, render_mode=None):
        super().__init__()

        '''
        Action space is represented by the vector: [Stright, right, left]
        example: [0, 1, 0] means the snake is moving to the right of its current direction
        '''
        self.action_space = spaces.Discrete(3)

        '''
        observation space is the following vector:
        [
            danger_stright,
            danger_right,
            danger_left,
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            food_left,
            food_right,
            food_up,
            food_down
        ]
        '''

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        #env parameters
        self.width = width
        self.height = height
        self.render_mode = render_mode

        #initialization
        pygame.init()
        if hasattr(pygame, 'font'):
            self.font = pygame.font.Font('arial.ttf', 25)
        else:
            self.font = pygame.font.SysFont('arial', 25)

        if self.render_mode == 'human':
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake GYM')
            self.clock = pygame.time.Clock()
        else:
            self.display = pygame.Surface((self.width, self.height))
            self.clock = None
        
        self.reset()
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
    
        self.direction = Direction.RIGHT

        self.head = Point(self.width/2, self.height/2)
        self.snake = [
            self.head,
            Point(self.head.x-BLOCK_SIZE, self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        obs = self.get_obs()
        info = self.get_info()

        return obs, info

    def step(self, action):
        self.frame_iteration += 1

        if self.render_mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        action_vect = np.zeros(3)
        action_vect[action] = 1

        self._move(action_vect)
        self.snake.insert(0, self.head)

        reward = 0
        done = False 

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
        elif self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        if self.render_mode in ['human', 'rgb_array']:
            self._update_ui()
            if self.render_mode == 'human':
                self.clock.tick(SPEED)
        
        obs = self.get_obs()
        info = self.get_info()

        return obs, reward, done, info

    def get_obs(self):
        # Create observation for reinforcement learning agent
        head = self.head
        
        # Points in all 4 directions from the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Current direction as a boolean
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # Determine danger based on current direction
        # Check if moving in current direction would result in collision
        danger_straight = False
        danger_right = False
        danger_left = False
        
        if dir_r:
            danger_straight = self.is_collision(point_r)
            danger_right = self.is_collision(point_d)
            danger_left = self.is_collision(point_u)
        elif dir_l:
            danger_straight = self.is_collision(point_l)
            danger_right = self.is_collision(point_u)
            danger_left = self.is_collision(point_d)
        elif dir_u:
            danger_straight = self.is_collision(point_u)
            danger_right = self.is_collision(point_r)
            danger_left = self.is_collision(point_l)
        elif dir_d:
            danger_straight = self.is_collision(point_d)
            danger_right = self.is_collision(point_l)
            danger_left = self.is_collision(point_r)
        
        # Create observation array
        obs = np.array([
            # Danger
            danger_straight,
            danger_right,
            danger_left,
            
            # Direction
            dir_l,
            dir_r, 
            dir_u,
            dir_d,
            
            # Food location relative to head
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y   # food down
        ], dtype=np.float32)
        
        return obs
    
    def get_info(self):
        return {
            'score': self.score,
            'snake_length': len(self.snake)
        }
    
    def render(self):
        if self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.display)), 
                axes=(1, 0, 2)
            )
    
    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x > self.width - BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False
    
    def _place_food(self):
        x = random.randint(0, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # Display score
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        
        if self.render_mode == 'human':
            pygame.display.flip()
    
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)

def main():
    env = SnakeEnv(render_mode='human')
    obs, info = env.reset()
    
    done = False
    while not done:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        
        if done:
            print(f"Game over! Final score: {info['score']}")
            env.reset()
    
    env.close()

if __name__ == "__main__":
    main()