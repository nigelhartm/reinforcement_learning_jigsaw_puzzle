import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from jigsaw import jigsaw_game

class NeuralNetwork(nn.Module):
    INPUTSIZE = 4*10
    ACTIONS = 25

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = self.ACTIONS
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.25
        self.number_of_iterations = 100000
        self.replay_memory_size = 20000
        self.minibatch_size = 500
        self.fc1 = nn.Linear(self.INPUTSIZE, 640)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(640, 1280)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc21 = nn.Linear(1280, 1280)
        self.relu21 = nn.ReLU(inplace=True)
        self.fc22 = nn.Linear(1280, 640)
        self.relu22 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(640, 320)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(320, 160)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(160, self.number_of_actions)
    def forward(self, x):
        out = x.view(x.size()[0], -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc21(out)
        out = self.relu21(out)
        out = self.fc22(out)
        out = self.relu22(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, start):
    solved_cnt = 0
    stats_reward = 0
    stats_file = open("pretrained_model/stats.tsv", "w")

    stats_file.write("1000its\tmean_rewards\n")
    stats_file.flush()

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    replay_memory = []
    epsilon = model.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    while iteration < model.number_of_iterations:
        list_state = []
        list_action = []
        list_state1 = []
        end_reward = 0
        game_state = jigsaw_game()
        init_action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        state_reward = game_state.get_state(init_action)
        reward = state_reward[1]
        state= torch.from_numpy(state_reward[0].astype(np.float32)).unsqueeze(0)
        finished = state_reward[2]
        while finished == False and reward > -1:
            state=state.unsqueeze(0)
            if torch.cuda.is_available():
                state = state.cuda()
            output = model(state)[0]
            mask = game_state.getMask().cuda()
            output = torch.sub(output, mask)
            #print(output)
            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():
                action = action.cuda()
            random_action = random.random() <= epsilon
            #if random_action:
                #print("Performed random action!")
            mask_copy = mask.cpu()
            mask_copy = mask_copy.numpy()
            action_space = np.sum(mask_copy == 0)
            rand_action = random.randint(1, action_space)
            cond = mask_copy == 1
            counts = np.cumsum(cond)
            idx = np.searchsorted(counts, rand_action)-1
            rand_action_new= np.zeros(1)
            rand_action_new[0] = idx
            rand_action_new = torch.from_numpy(rand_action_new)
            rand_action_new = rand_action_new.type(torch.int)
            action_index = [rand_action_new
                            if random_action
                            else torch.argmax(output)][0]
            if torch.cuda.is_available():
                action_index = action_index.cuda()
            action[action_index] = 1
            action = action.cpu()
            state_reward = game_state.get_state(action)
            state_1 = torch.from_numpy(state_reward[0].astype(np.float32)).unsqueeze(0)
            reward = state_reward[1]
            finished = state_reward[2]
            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
            end_reward = reward
            list_state.append(state)
            list_action.append(action)
            list_state1.append(state_1)

            state = state_1
        for i in range(0, len(list_state)):
            replay_memory.append((list_state[i], list_action[i], end_reward, list_state1[i], finished))
        # if replay memory is full, remove the oldest transitions
        while len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()
        state_1_batch = state_1_batch.unsqueeze(1)
        output_1_batch = model(state_1_batch)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        #print("loss:" + str(loss))
        optimizer.step()
        state = state_1
        iteration += 1
        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")
        if(end_reward <= 0):
            stats_reward += 0
        else:
            stats_reward += int(end_reward)
        if iteration % 1000 == 0:
            stats_file.write(str(iteration) + "\t" + str(float(stats_reward/1000)) + "\n")
            stats_file.flush()
            stats_reward = 0
        if finished:
            solved_cnt += 1
        print("iteration:", iteration, "\telapsed time:", time.time()-start, "\tepsilon:", epsilon, "\taction:",
              action_index.cpu().detach().numpy(), "\treward:", reward.numpy()[0][0], "\tQ max:",
              np.max(output.cpu().detach().numpy()), "\tSolved puzzles:", solved_cnt)
    stats_file.close()

def main(mode):
    cuda_is_available = torch.cuda.is_available()
    if mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
        model = NeuralNetwork()
        if cuda_is_available:
            model = model.cuda()
        model.apply(init_weights)
        start = time.time()
        train(model, start)

if __name__ == "__main__":
    main(sys.argv[1])