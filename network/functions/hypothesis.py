import numpy as np

from network.functions import reform
import config

def sigmoid(z):
    #print(-z)
    return 1.0 / (1.0 + np.exp(-z.astype(np.longdouble)))

def hypothesis(t, x, placeholder=0, computingGradient = False, printData = False):

    if isinstance(t,np.ndarray) and np.shape(t)[0] == 1:
        print("yes")
        t = reform.reform_theta(t) #for bfgs, whose theta input is flattened

    # print("hypothesis data:")
    # print(np.shape(x))
    # print(np.shape(t))

    aList = []
    zList = []

    x = np.insert(x, 0, 1, axis=0) #Add bias unit to input
    aList.append(x) #a0
    a = x
    for i in range(config.network_size[1]):

        z = np.dot(t[i], a)
        a = sigmoid(z)
        a = np.insert(a, 0, 1, axis=0) #Add bias unit to a

        if computingGradient:
            zList.append(z) #z0
            aList.append(a) #a1

    h = np.dot(t[config.network_size[1]], a)
    if computingGradient:
        zList.append(h) #z2

    h = sigmoid(h)
    if computingGradient:
        aList.append(h) #a3

    if computingGradient:
        return h, aList, zList
    else:
        return h
