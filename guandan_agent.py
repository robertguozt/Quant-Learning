import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class GuandanEnv:
    def __init__(self, level_rank):
        self.level_rank = level_rank
        self.hand = [0] * 15
        self.wild_card_count = 0

    def add_card(self, rank, suit):
        if rank == self.level_rank and suit == 1:
            self.wild_card_count += 1
        else:
            index = rank - 2 
            self.hand[index] += 1

    def get_state(self):
        return np.array(self.hand + [self.wild_card_count], dtype=np.float32)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)
class QAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.input_dim = 16
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.model = PolicyNetwork(self.input_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            
            state_tensor = torch.FloatTensor(state)
            predicted_q = self.model(state_tensor)
            
            target_f = predicted_q.clone().detach()
            target_f[action] = target
            
            # Optimization step
            loss = self.loss_fn(predicted_q, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_ai():
    total_actions = 17 
    agent = QAgent(total_actions)
    
    print("🚀 Starting Guandan AI Training Session...")

    for episode in range(5000):
        game = GuandanEnv(level_rank=5)
        for _ in range(27):
            game.add_card(random.randint(2, 16), random.randint(0, 3))
        
        total_reward = 0
        done = False
        
        for step in range(100):
            current_state = game.get_state()
            action = agent.select_action(current_state)
            
            reward = 0
            if action < 15 and game.hand[action] > 0:
                game.hand[action] -= 1
                reward = 10 
            elif action == 15 and game.wild_card_count > 0:
                game.wild_card_count -= 1
                reward = 20 
            else:
                reward = -10 
            
            # Check if hand is empty
            done = (sum(game.hand) + game.wild_card_count == 0)
            if done:
                reward += 100
            
            next_state = game.get_state()
            agent.remember(current_state, action, reward, next_state, done)
            total_reward += reward
            
            if done:
                break
        
        agent.replay(32)
        
        if episode % 50 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Exploration Rate: {agent.epsilon:.2f}")

    print("Training Complete")

if __name__ == "__main__":
    train_ai()
