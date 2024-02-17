
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from jigsaw import jigsaw_game

class NeuralNetwork(nn.Module):
    INPUTSIZE = 4*10
    ACTIONS = 25

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = self.ACTIONS
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.50
        self.number_of_iterations = 100000
        self.replay_memory_size = 1000000
        self.minibatch_size = 3000
        self.fc1 = nn.Linear(self.INPUTSIZE, 2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2048, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, 2048)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2048, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc42 = nn.Linear(1024, 1024)
        self.relu42 = nn.ReLU(inplace=True)
        self.fc44 = nn.Linear(1024, 512)
        self.relu44 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)
    def forward(self, x):
        out = x.view(x.size()[0], -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc42(out)
        out = self.relu42(out)
        out = self.fc44(out)
        out = self.relu44(out)
        out = self.fc5(out)
        return out

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, start):
    # Statistics
    solved_cnt = 0
    stats_reward = 0

    # Setup Learning
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()
    replay_memory = []
    epsilon = model.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # Iterate training
    while iteration < model.number_of_iterations:
        list_state = []
        list_action = []
        list_state1 = []

        # Init Game
        end_reward = 0
        game_state = jigsaw_game()
        init_action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        state_reward = game_state.get_state(init_action)
        reward = state_reward[1]
        state= torch.from_numpy(state_reward[0].astype(np.float32)).unsqueeze(0)
        step = state_reward[3]
        finished = state_reward[2]

        # Play until solved or to much steps
        while finished == False:
            state=state.unsqueeze(0)
            if torch.cuda.is_available():
                state = state.cuda()
            output = model(state)[0]
            mask = game_state.getMask().cuda()
            output = torch.sub(output, mask)
            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():
                action = action.cuda()
            random_action = random.random() <= epsilon
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
            step = state_reward[3]
            action = action.unsqueeze(0)
            reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
            end_reward = reward
            list_state.append(state)
            list_action.append(action)
            list_state1.append(state_1)
            state = state_1
        
        for i in range(0, len(list_state)):
            replay_memory.append((list_state[i], list_action[i], end_reward, list_state1[i], finished))
        while len(replay_memory) > model.replay_memory_size:
            x = random.randint(0, len(replay_memory)-1)
            replay_memory.pop(x)
        # sample
        epsilon = epsilon_decrements[iteration]
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        # just update them?
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()
        state_1_batch = state_1_batch.unsqueeze(1)
        output_1_batch = model(state_1_batch)
        y_batch = torch.cat(tuple(reward_batch[i]
                                  for i in range(len(minibatch))))
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)
        optimizer.zero_grad()
        y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()
        state = state_1
        iteration += 1
        stats_reward += int(end_reward)

        # Save every 10000 iterations
        if iteration % 10000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")
        if finished and end_reward>0:
            solved_cnt += 1
        wandb.log({"iteration": iteration, "epsilon": epsilon, "reward": reward.numpy()[0][0], "qmax": np.max(output.cpu().detach().numpy()), "solved": solved_cnt, "solved_per_iteration": solved_cnt/iteration, "reward_per_iteration": reward/iteration})

# Main function
def main():
    # start a new wandb run to track this script
    wandb.init(
        project="jigsaw_dqn",
        config={
        "test": "jigsaw",
        }
    )
    if not os.path.exists('pretrained_model/'):
        os.mkdir('pretrained_model/')
    model = NeuralNetwork()
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        sys.exit("No cuda gpu available.")
    model.apply(init_weights)
    start = time.time()
    train(model, start)
    wandb.finish()

if __name__ == "__main__":
    main()