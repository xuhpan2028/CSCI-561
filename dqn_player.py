import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from copy import deepcopy

class DQN(nn.Module):
    def __init__(self, board_size, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 input channels: own, opponent, empty
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x):
        # x shape: batch_size x 3 x board_size x board_size
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Output shape: batch_size x action_size

class DQNPlayer:
    def __init__(self, board_size=5, action_size=26):
        self.type = 'dqn'
        self.board_size = board_size
        self.action_size = action_size
        self.epsilon = 1.0  # For epsilon-greedy policy
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        self.model = DQN(board_size, action_size)
        self.target_model = DQN(board_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.steps = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def get_valid_actions(self, go, piece_type):
        valid_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    valid_actions.append((i, j))
        valid_actions.append("PASS")
        return valid_actions

    def select_action(self, state, go, piece_type):
        valid_actions = self.get_valid_actions(go, piece_type)
        if np.random.rand() <= self.epsilon:
            action = random.choice(valid_actions)
        else:
            # Use model to predict Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                q_values = self.model(state_tensor)
            q_values = q_values.cpu().numpy()[0]
            # Mask invalid actions
            action_indices = self.get_action_indices(valid_actions)
            masked_q_values = np.full(self.action_size, -np.inf)
            for idx in action_indices:
                masked_q_values[idx] = q_values[idx]
            action_idx = np.argmax(masked_q_values)
            action = self.get_action_from_index(action_idx)
        return action

    def get_action_indices(self, valid_actions):
        # Map valid actions to indices
        indices = []
        for action in valid_actions:
            if action == "PASS":
                indices.append(self.action_size - 1)
            else:
                i, j = action
                indices.append(i * self.board_size + j)
        return indices

    def get_action_from_index(self, index):
        if index == self.action_size - 1:
            return "PASS"
        else:
            i = index // self.board_size
            j = index % self.board_size
            return (i, j)

    def get_action_index(self, action):
        if action == "PASS":
            return self.action_size - 1
        else:
            i, j = action
            return i * self.board_size + j

    def get_input(self, go, piece_type):
        # Get current state
        state = self.get_state(go, piece_type)
        # Select action
        action = self.select_action(state, go, piece_type)
        return action

    def get_state(self, go, piece_type):
        # Convert board to state representation
        # For each position, we have 3 channels: own stones, opponent stones, empty
        board = np.array(go.board)
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state[0][board == piece_type] = 1  # Own stones
        state[1][board == 3 - piece_type] = 1  # Opponent stones
        state[2][board == 0] = 1  # Empty
        return state

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor([e[0] for e in batch])
        action_batch = [e[1] for e in batch]
        reward_batch = torch.FloatTensor([e[2] for e in batch])
        next_state_batch = torch.FloatTensor([np.zeros_like(e[0]) if e[3] is None else e[3] for e in batch])
        done_batch = torch.FloatTensor([e[4] for e in batch])

        # Compute target Q-values
        with torch.no_grad():
            target_q_values = self.target_model(next_state_batch)
            max_target_q_values, _ = torch.max(target_q_values, dim=1)
            target = reward_batch + self.gamma * max_target_q_values * (1 - done_batch)
        # Compute current Q-values
        q_values = self.model(state_batch)
        action_indices = [self.get_action_index(a) for a in action_batch]
        q_values = q_values.gather(1, torch.tensor(action_indices).unsqueeze(1)).squeeze(1)
        # Compute loss
        loss = nn.functional.mse_loss(q_values, target)
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Update target network periodically
        self.steps += 1
        if self.steps % 1000 == 0:
            self.update_target_model()