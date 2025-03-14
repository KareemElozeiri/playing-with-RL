import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import os 


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x  = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)

        return x
    
    def save(self, file_name="dqn.pt"):
        dir = './model'
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        file_name = os.path.join(dir, file_name)
        torch.save(self.state_dict(), file_name)


class DQNTraniner:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )
        
        # Predict Q values with current state
        Q_old = self.model(state)
        target = Q_old.clone()

        for idx in range(len(done)):
            Q_new = reward[idx] 
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Fix: use the correct action index for this specific sample
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, Q_old)
        loss.backward()

        self.optimizer.step()