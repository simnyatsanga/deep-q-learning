from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from dqn import QLearner, compute_td_loss, ReplayBuffer

USE_CUDA = torch.cuda.is_available()
USE_FRAME_STACK = False
env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env, frame_stack=USE_FRAME_STACK)
env = wrap_pytorch(env)

train_num_frames = 5000000
sample_num_frames = 50000
batch_size = 32
gamma = 0.99
target_update = 50000
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 1000000
replay_initial = 10000
learning_rate = 1e-5
train_replay_buffer = ReplayBuffer(100000)
analysis_replay_buffer = ReplayBuffer(100000)

policy_model = QLearner(env, train_num_frames, batch_size, gamma, train_replay_buffer)
target_model = QLearner(env, train_num_frames, batch_size, gamma, train_replay_buffer)
target_model.load_state_dict(policy_model.state_dict())
target_model.eval()

optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if USE_CUDA:
    policy_model = policy_model.to(device)
    target_model = target_model.to(device)

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

def play_to_train(num_frames, policy_model, target_model, buffer):
    losses = []
    all_rewards = []
    mean_losses = []
    mean_rewards = []
    episode_reward = 0
    state = env.reset()

    start_training = time.time()
    for frame_idx in range(1, num_frames + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = policy_model.act(state, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
        
        if len(buffer) > replay_initial:
            loss = compute_td_loss(policy_model, target_model, batch_size, gamma, buffer, device)
            optimizer.zero_grad()
            loss.backward()
            for param in policy_model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            losses.append(loss.data.cpu().numpy())

        if frame_idx % 10000 == 0 and len(buffer) <= replay_initial:
            print('#Frame: %d, preparing replay buffer' % frame_idx)

        if frame_idx % 10000 == 0 and len(buffer) > replay_initial:
            mean_losses.append(np.mean(losses))
            print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses)))
            mean_rewards.append(np.mean(all_rewards[-10:]))
            print('Last-10 average reward: %f' % np.mean(all_rewards[-10:]))
        
        # Update the target network, copying all weights and biases in DQN
        if frame_idx % target_update == 0:
            target_model.load_state_dict(policy_model.state_dict())
        
        # Saving checkpoints after every million frames
        if frame_idx % 1000000 == 0:
            model_filename = "dqn_pong_model_%s" % (frame_idx)
            torch.save(policy_model.state_dict(), model_filename)

    end_training = time.time()

    print(f'Total training time - {(end_training - start_training) / 3600} hours')

    # Save all mean losses
    with open('mean_losses.npy', 'wb') as losses_file:
        np.save(losses_file, np.array(mean_losses))

    # Save all mean rewards
    with open('mean_rewards.npy', 'wb') as rewards_file:
        np.save(rewards_file, np.array(mean_rewards))

    # Save the final policy model
    torch.save(policy_model.state_dict(), "dqn_pong_model_final")

def play_to_sample(num_frames, model, buffer):
    state = env.reset()
    for frame_idx in range(1, num_frames + 1):

        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        if done:
            state = env.reset()

print('Start Training ...')
play_to_train(train_num_frames, policy_model, target_model, train_replay_buffer)
print('Done Training ...')

print('Playing before sampling ...')
play_to_sample(sample_num_frames, policy_model, analysis_replay_buffer)


# Sample 1000 frames from the sample replay buffer for analysis
print('Sampling from the analysis replay buffer ...')
states, actions, rewards, next_states, done = analysis_replay_buffer.sample(1000)

with open('sampled_states.npy', 'wb') as states_f:
    np.save(states_f, states)

with open('sampled_actions.npy', 'wb') as actions_f:
    np.save(actions_f, np.array(actions))

with open('sampled_rewards.npy', 'wb') as rewards_f:
    np.save(rewards_f, np.array(rewards))

with open('sampled_next_states.npy', 'wb') as next_states_f:
    np.save(next_states_f, next_states)

with open('sampled_done.npy', 'wb') as done_f:
    np.save(done_f, np.array(done))

