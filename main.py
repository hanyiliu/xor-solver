import numpy as np

from network.training import train_network
from network.functions import reform
from network.functions import hypothesis
import config

input = np.reshape(np.genfromtxt(config.input_dir),(8,3))
output = np.reshape(np.genfromtxt(config.output_dir),(8,2))

train_network.train_network(input,output)


theta = reform.reform_theta(np.genfromtxt(config.theta_dir))

h0 = hypothesis.hypothesis(theta, input[0])
h1 = hypothesis.hypothesis(theta, input[1])
h2 = hypothesis.hypothesis(theta, input[2])
h3 = hypothesis.hypothesis(theta, input[3])
h4  = hypothesis.hypothesis(theta, input[4])
