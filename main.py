import numpy as np
import sys
import os
import time

from network.training import train_network
from network.training import vectorized_train_network
from network.functions import reform
from network.functions import hypothesis
from network.functions import computeGradient
import config

np.set_printoptions(edgeitems=30, linewidth=1000,
    formatter=dict(float=lambda x: "%.3g" % x))

np.set_printoptions(suppress=True)

input = np.reshape(np.genfromtxt(config.input_dir),(8,3))
output = np.reshape(np.genfromtxt(config.output_dir),(8,2))
#
# sys.stdout = open(os.devnull, 'w')
#
print("beginning UNVECTORIZED training:")
unvectorized_start = time.time()
train_network.train_network(input,output)
unvectorized_end = time.time()
print("UNVECTORIZED training finished, took: {} seconds".format(unvectorized_end-unvectorized_start))

print("beginning VECTORIZED training:")
vectorized_start = time.time()
vectorized_train_network.train_network(input,output)
vectorized_end = time.time()
print("VECTORIZED training finished, took: {} seconds".format(vectorized_end-vectorized_start))
#
# sys.stdout = sys.__stdout__

theta = reform.reform_theta(np.genfromtxt(config.theta_dir))

# print("single hypotheses:")
#
# h0 = hypothesis.hypothesis(theta, input[0])
# h1 = hypothesis.hypothesis(theta, input[1])
# h2 = hypothesis.hypothesis(theta, input[2])
# h3 = hypothesis.hypothesis(theta, input[3])
# h4  = hypothesis.hypothesis(theta, input[4])
# print("{}, {}, {}, {}, {}".format(h0,h1,h2,h3,h4))
#
# print("vectorized hypotheses:")
# h = hypothesis.hypothesis(theta, input)
# print("{}".format(h))

# print("single gradients:")
#
# gradient0 = computeGradient.computeGradient(theta, input[0], output[0])
# gradient1 = computeGradient.computeGradient(theta, input[1], output[1])
# gradient2 = computeGradient.computeGradient(theta, input[2], output[2])
# gradient3 = computeGradient.computeGradient(theta, input[3], output[3])
# gradient4 = computeGradient.computeGradient(theta, input[4], output[4])
# gradient5 = computeGradient.computeGradient(theta, input[5], output[5])
# gradient6 = computeGradient.computeGradient(theta, input[6], output[6])
# gradient7 = computeGradient.computeGradient(theta, input[7], output[7])
#
# #print("gradient0: {} \n\n {} \n\n {} \n\n {} \n\n".format(gradient0[0],gradient0[1],gradient0[2],gradient0[3]))
# print("gradient1: {} \n\n {} \n\n {} \n\n {} \n\n".format(gradient1[0],gradient1[1],gradient1[2],gradient1[3]))
# print("gradient2: {} \n\n {} \n\n {} \n\n {} \n\n".format(gradient2[0],gradient2[1],gradient2[2],gradient2[3]))
# # print("gradient3: {} \n\n {} \n\n {} \n\n {} \n\n".format(gradient3[0],gradient3[1],gradient3[2],gradient3[3]))
# # print("gradient4: {} \n\n {} \n\n {} \n\n {} \n\n".format(gradient4[0],gradient4[1],gradient4[2],gradient4[3]))
# single_gradient_sum = gradient0
# #print("single gradient sum:\n {} \n\n {} \n\n {} \n\n {} \n\n".format(single_gradient_sum[0],single_gradient_sum[1],single_gradient_sum[2],single_gradient_sum[3]))
# for i in range(0,4):
#     single_gradient_sum[i] = single_gradient_sum[i] + gradient1[i] + gradient2[i] + gradient3[i] + gradient4[i] + gradient5[i] + gradient6[i] + gradient7[i]
#     #print("single gradient sum:\n {} \n\n {} \n\n {} \n\n {} \n\n".format(single_gradient_sum[0],single_gradient_sum[1],single_gradient_sum[2],single_gradient_sum[3]))
#     single_gradient_sum[i] = single_gradient_sum[i]/8
#     #print("single gradient sum:\n {} \n\n {} \n\n {} \n\n {} \n\n".format(single_gradient_sum[0],single_gradient_sum[1],single_gradient_sum[2],single_gradient_sum[3]))
#
# # single_gradient_sum = gradient0 + gradient1 + gradient2 + gradient3 + gradient4 + gradient5 + gradient6 + gradient7
# # single_gradient_sum = gradient1 + gradient2
# print("single gradient sum:\n {} \n\n {} \n\n {} \n\n {} \n\n".format(single_gradient_sum[0],single_gradient_sum[1],single_gradient_sum[2],single_gradient_sum[3]))
#
#
# print("vectorized gradients:")
# gradient = computeGradient.computeGradient(theta, input, output)
# print("gradient: \n {} \n\n {} \n\n {} \n\n {} \n\n".format(gradient[0],gradient[1],gradient[2],gradient[3]))
