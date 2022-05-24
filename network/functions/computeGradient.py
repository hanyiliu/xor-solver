import numpy as np

from network.functions import hypothesis
from network.functions import reform
import config

def sigmoid(z):
    #print(-z)
    return 1.0 / (1.0 + np.exp(-z.astype(np.longdouble)))
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z));

def computeGradient(t, x, y, flattenGradient=False): #For only one input and output right now

    # if isinstance(t,np.ndarray) and np.size(np.shape(t)) == 1 and not isinstance(t,list):
    #     print("gradient yes")
    #     t = reform.reform_theta(t) #for bfgs, whose theta input is flattened

    h,a,z = hypothesis.hypothesis(t, x, computingGradient=True) #Step 1

    deltaList = []


    #This goes backwards (back-propagation)
    delta = a[-1]-y #for output units
    deltaList.insert(0, delta) #delta 4

    for i in range(config.network_size[1]):
        j = config.network_size[1]-i
        z[j-1] = np.insert(z[j-1], 0, 1, axis=0) #add 0 to account for bias unit
        delta = np.dot(t[j].T,delta)*sigmoidGradient(z[j-1])
        delta = np.delete(delta, 0) #remove bias unit's delta
        deltaList.insert(0, delta) #delta 3

    gradientList = []
    for i in range(config.network_size[1]+1):
        gradientList.append(np.dot(np.array([deltaList[i]]).T,np.array([a[i]])))


    if flattenGradient:
        flatGradient = np.array(())
        for i in range(config.network_size[1]+1):
            flatGradient = np.concatenate((flatGradient.flatten(),gradientList[i].flatten()))
        return flatGradient
    else:
        return gradientList
