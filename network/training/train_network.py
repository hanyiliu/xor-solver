import numpy as np
import config

from network.functions import cost
from network.functions import hypothesis
from network.functions import computeGradient
from network.functions import reform

def train_network(x, y): #player = 0,1

    m = np.shape(x)[0]

    #[5,2]
    if config.random_theta:
        t = np.random.uniform(-1, 1, (config.network_size[1]+1,config.network_size[0],config.network_size[0]+1)) #[# of layers, # of neurons, # of neurons + 1 to account for bias unit]
        print(np.shape(t))
        thetaList = []
        for i in range(0, np.shape(t)[0]):
            if i == 0: #Fix first layer to account for input units
                thetaList.append(t[0,:,0:config.input+1])
            elif i == np.shape(t)[0]-1: #Fix last layer to account for output units
                thetaList.append(t[-1,0:config.output])
            else:
                thetaList.append(t[i])

        t = thetaList
    else:
        t = reform.reform_theta(np.genfromtxt(config.theta_dir))

    costs = []
    c = 0 #cost
    g = [] #gradient
    for i in range(len(t)):
        g.append(np.zeros(np.shape(t[i])))
    for j in range(0,config.iterations):

        for i in range(0, np.shape(x)[0]):
            #print("data: {}".format(i))
            #print("training data {}".format(i))
            hypothesis0 = hypothesis.hypothesis(t, x[i])
            gradient = computeGradient.computeGradient(t, x[i], y[i])


            cost0 = cost.cost(hypothesis0,y[i])
            #either place epsilon here or place at t - g

            for i in range(len(g)):
                g[i] = g[i] + gradient[i]


            costs.append(cost0)
            c = cost0

        for i in range(len(t)):
            temp = np.copy(t[i])
            temp[:,0] = 0
            t[i] = t[i]-(g[i]/m + config.lamb*temp/m) #adds regularization

        g = [] #gradient
        for i in range(len(t)):
            g.append(np.zeros(np.shape(t[i])))
        #print("finished iteration {} of {}. cost: {}%".format(j+1, config.iterations, c))

    #print("finished, saving thetas")

    flatThetas = np.array(())
    for i in range(len(t)):
        flatThetas = np.append(flatThetas, t[i])

    np.savetxt(config.theta_dir, flatThetas) #Overwrites current theta values. TODO: fix numpy conversion


    #print("overall cost improvement: {}%".format(100*abs(costs[0]-costs[-1])/((costs[0]+costs[-1])/2)))
