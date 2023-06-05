# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
#
# conda create -n test41
# conda actiate test41
#  conda install -c conda-forge tensorflow
#  conda install -c conda-forge gym
# conda install -c cogsci pygame 
#  conda install -c conda-forge wandb 
#
# git clone git@github.com:tensorneko/keras-rl2.git
# cd keras-rl2
# python setup.py install
#
# export
# conda env export > environment_droplet.yml
# import
# conda env create -f environment.yml
# list all the conda environment available
# conda info --envs
# Create new environment named as `envname`
# conda create --name envname
# Remove environment and its dependencies
# conda remove --name envname --all
# Clone an existing environment
# conda create --name clone_envname --clone envname
#
#
#
# vs_code -> python select interpreter -> conda test

###https://www.gymlibrary.dev/content/environment_creation/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import gym
import numpy as np
