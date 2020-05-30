from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 10000000
batch_size = 32
gamma = 0.99
target_update = 1000
    
replay_initial = 10000
replay_buffer = ReplayBuffer(1000000)

policy_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
target_model.load_state_dict(policy_model.state_dict())
target_model.eval()

optimizer = optim.Adam(policy_model.parameters(), lr=0.00001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if USE_CUDA:
    policy_model = policy_model.cuda()
    target_model = target_model.cuda()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()


for frame_idx in range(1, num_frames + 1):

    epsilon = epsilon_by_frame(frame_idx)
    action = policy_model.act(state, epsilon)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(policy_model, target_model, batch_size, gamma, replay_buffer, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.cpu().numpy())

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial:
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))
    
    # Update the target network, copying all weights and biases in DQN
    if frame_idx % target_update == 0:
        print("Updating target_model parameters with policy_model parameters.")
        target_model.load_state_dict(policy_model.state_dict())

torch.save(policy_model.state_dict(), "dqn_pong_model")