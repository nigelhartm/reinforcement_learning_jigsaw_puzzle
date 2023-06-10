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

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = 25
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(1, 32, 2, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 2, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, 2, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(7, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        #print("Froward:")
        #print(x.size())
        out = self.conv1(x)
        #print(out.size())
        out = self.relu1(out)
        out = self.conv2(out)
        #print(out.size())
        out = self.relu2(out)
        out = self.conv3(out)
        #print(out.size())
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        #print(out.size())
        out = self.fc4(out)
        #print(out.size())
        out = self.relu4(out)
        out = self.fc5(out)
        #print(out.size())
        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, start):
    solved_cnt = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss() # initialize mean squared error loss
    game_state = jigsaw_game() # instantiate game
    replay_memory = [] # initialize replay memory

    # initial action is get a new piece
    action = np.array([0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0,
                       1])
    game_state.action_converter(action)
    state_reward = game_state.get_state()
    state= torch.from_numpy(state_reward[0].astype(np.float32)).unsqueeze(0)
    reward = state_reward[1]
    finished = state_reward[2]

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]
        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        game_state.action_converter(action)
        state_reward = game_state.get_state()
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
        minibatch = random.sample(replay_memory, 1) #min(len(replay_memory), model.minibatch_size))???????????????????????????????

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

        print(state_1_batch.unsqueeze(0))
        # get output for the next state
        output_1_batch = model(state_1_batch) #.unsqueeze(0))??????????????????????????????????????????????????????????????????????????

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
        if finished:
            solved_cnt=solved_cnt+1
        print("iteration:", iteration, "elapsed time:", time.time()-start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()), "Solved puzzles:", solved_cnt, "\n")

"""
def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1
"""

def main(mode):
    cuda_is_available = torch.cuda.is_available()

    """if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model)"""
    #elif
    if mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main(sys.argv[1])
