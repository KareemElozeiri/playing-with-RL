import torch 
import random 
import numpy as np
from collections import deque
from game import SnakeEnv, Direction, Point
from model import DQN, DQNTraniner
from utils import plot

MAX_MEM = 100000
BATCH_SIZE = 1000
lr = 0.001



class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon  = 0 #prob for random action
        self.gamma =    0.9 # dicount factor
        self.mem = deque(maxlen=MAX_MEM)

        self.model = DQN(11, 128, 3) 
        self.trainer = DQNTraniner(self.model, lr=lr, gamma=self.gamma) 

    def get_state(self, game):
        return game.get_obs()
    

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(80 - self.n_games, 5)
        final_move = [0, 0, 0]
        
        # Lower random threshold to 100 instead of 200
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def remember(self, state, action, reward, next_state, done):
        self.mem.append((state, action, reward, next_state, done)) 


    def train_long_mem(self):
        if len(self.mem) == 0:
            return  # Skip training if memory is empty
            
        if len(self.mem) > BATCH_SIZE:
            mini_sample = random.sample(self.mem, BATCH_SIZE)
        else:
            mini_sample = self.mem
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

        


    def train_short_mem(self, state, action, reward, new_state, done):
        self.trainer.train_step(state, action, reward, new_state, done) 
 




def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeEnv(render_mode='human')
    obs, _ = game.reset()

    while True:
        old_state = agent.get_state(game)
        action = agent.get_action(old_state)

        obs, reward, done, info = game.step(action.index(1))

        new_state = agent.get_state(game)

        agent.train_short_mem(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)  

        if done:
            game.reset()
            agent.n_games +=1
            agent.train_long_mem()
            
            if info['score'] > record:
                record = info['score']
                agent.model.save()
            
            print(f"Game: {agent.n_games} | Score: {info['score']}")
            scores.append(info['score'])
            total_score += info['score']
            mean_scores.append(total_score/agent.n_games)

            plot(scores=scores, mean_scores=mean_scores)


if __name__ == "__main__":
    train()






