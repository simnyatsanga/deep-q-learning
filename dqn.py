from collections import deque
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math, random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device) if USE_CUDA else autograd.Variable(*args, **kwargs)

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon: # Exploit
            state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            ######## YOUR CODE HERE! ########
            # TODO: Given state, you should write code to get the Q value and chosen action
            # Complete the R.H.S. of the following 2 lines and uncomment them
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.max(1)[1].view(1, 1).item()
            ######## YOUR CODE HERE! ########
        else: # Explore
            action = random.randrange(self.env.action_space.n)
        return action
        
def compute_td_loss(policy_model, target_model, batch_size, gamma, replay_buffer, device):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is False, done)), device=device, dtype=torch.bool)
    non_final_next_states = []
    for i, nx_state in enumerate(next_state):
        if done[i] == False:
            non_final_next_states.append(nx_state)
    
    non_final_next_states = Variable(torch.FloatTensor(np.float32(non_final_next_states)))
    state_batch = Variable(torch.FloatTensor(np.float32(state)))
    next_state_batch = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action_batch = Variable(torch.LongTensor(action)).reshape(batch_size, 1)
    reward_batch = Variable(torch.FloatTensor(reward))
    done_batch = Variable(torch.FloatTensor(done))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each state according to the policy_model
    state_action_values = policy_model.forward(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_model.forward(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values: r + gamma * max_a(Q(s', a))
    expected_state_action_values = reward_batch + (gamma * next_state_values)

    ######## YOUR CODE HERE! ########
    # TODO: Implement the Temporal Difference Loss
    
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # mse_loss = nn.MSELoss()
    # loss = mse_loss(state_action_values, expected_state_action_values)
    ######## YOUR CODE HERE! ########
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        ######## YOUR CODE HERE! ########
        # TODO: Randomly sampling data with specific batch size from the buffer
        # Hint: you may use the python library "random".
        # If you are not familiar with the "deque" python library, please google it.
        ######## YOUR CODE HERE! ########
        batch = random.sample(self.buffer, batch_size)
        state = [tup[0] for tup in batch]
        action = [tup[1] for tup in batch]
        reward = [tup[2] for tup in batch]
        next_state = [tup[3] for tup in batch]
        done = [tup[4] for tup in batch]
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)
