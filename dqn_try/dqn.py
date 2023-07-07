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
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 128
        self.fc1 = nn.Linear(self.INPUTSIZE, 80)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(80, 80)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(80, 40)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(40, self.number_of_actions)
    def forward(self, x):
        out = x.view(x.size()[0], -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc6(out)
        return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, start):
    stats_file = open("pretrained_model/stats.tsv", "w")
    puzzlestats_file = open("pretrained_model/puzzle_stats.tsv", "w")

    puzzlestats_file.write("puzzle\tactions\n")
    puzzlestats_file.flush()

    stats_file.write("iteration\trewards\n")
    stats_file.flush()

    solved_cnt = 0
    action_cnt = 0
    iter_reward = 0

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss() # initialize mean squared error loss
    game_state = jigsaw_game() # instantiate game
    replay_memory = [] # initialize replay memory

    # initial action is get a new piece
    init_action = np.array([0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0,
                       1])
    state_reward = game_state.get_state(init_action)
    state= torch.from_numpy(state_reward[0].astype(np.float32)).unsqueeze(0)
    reward = state_reward[1]
    finished = state_reward[2]

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main infinite loop
    while iteration < model.number_of_iterations:
        state=state.unsqueeze(0)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state = state.cuda()
        output = model(state)[0]
        mask = game_state.getMask().cuda()
        output = torch.mul(output, mask) # mask the output to valid moves
        print("Masked output:")
        print(output)

        # initialize action
        action_cnt+=1
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        #action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
        #                if random_action
        #                else torch.argmax(output)][0]
        # THIS PART IS TRICKY but basically we just try to map the random action to masked actions
        #
        mask_copy = mask.cpu()
        mask_copy = mask_copy.numpy()
        action_space = np.sum(mask_copy == 1)
        rand_action = random.randint(1, action_space)
 
        cond = mask_copy == 1
        counts = np.cumsum(cond)
        idx = np.searchsorted(counts, rand_action)

        rand_action_new= np.zeros(1)
        rand_action_new[0] = idx
        rand_action_new = torch.from_numpy(rand_action_new)
        rand_action_new = rand_action_new.type(torch.int)
        print(rand_action_new)
        action_index = [rand_action_new
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        action = action.cpu()

        state_reward = game_state.get_state(action)
        state_1= torch.from_numpy(state_reward[0].astype(np.float32)).unsqueeze(0)
        reward = state_reward[1]
        finished = state_reward[2]

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, finished))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()
        
        state_1_batch = state_1_batch.unsqueeze(1)
        # get output for the next state
        output_1_batch = model(state_1_batch)
        
        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))
        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        iter_reward += reward
        ## print overall reward every 10000 steps
        if iteration % 10000 == 0:
            stats_file.write(str(iteration) + "\t" + str(int(iter_reward[0][0])) + "\n")
            stats_file.flush()
            iter_reward = 0
        if finished:
            solved_cnt=solved_cnt+1
            print("FINISHED:")
            print("in -> " + str(action_cnt))
            puzzlestats_file.write(str(solved_cnt) + "\t" + str(action_cnt) + "\n")
            puzzlestats_file.flush()
            action_cnt = 0
            state_reward = game_state.get_state(init_action)
            state= torch.from_numpy(state_reward[0].astype(np.float32)).unsqueeze(0)
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
