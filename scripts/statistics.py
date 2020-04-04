import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def calculate_moholonobis(data, mu, covariance):
    # data = np.array(      [x,y])
    # mu = np.array(        [mux,muy])
    # covariance = np.array([[x,xy],[xy,y]])
    diff = data-mu
    inv_cov = np.linalg.inv(covariance)
    moholonobis = diff.transpose().reshape(1,2)@inv_cov@(diff.reshape(2,1))

    # ??? maybe different
    # moholonobis = sp.linalg.sqrtm(moholonobis)
    moholonobis = np.sqrt(moholonobis)
    # ???

    return moholonobis

def calculate_moholonobis_vector(data, mu, covariance):
    # data = np.array(       [ [x1,y1],        [x2,y2]       ,...])  <-- ground truth
    # mu = np.array(         [ [mux1,muy1],    [mux2,muy2]   ,...])  <-- from nn
    # covariance = np.array([[ [x,xy],[xy,y]], [x,xy],[xy,y]],...])  <-- from nn
    mv = []
    for i in range(len(data)):
        mv.append(calculate_moholonobis(data[i], mu[i], covariance[i]))
    nmv = np.array(mv)
    return nmv

import math
def is_inside_sigma(data, mu, covariance,isok = 3):
    # data = np.array(      [x,y])
    # mu = np.array(        [mux,muy])
    # covariance = np.array([[x,xy],[xy,y]])
    # sigmas = [covariance[0][0],covariance[1][1]]
    
    sigmas = [isok * math.sqrt(covariance[0][0]),isok * math.sqrt(covariance[1][1])]

    xcheck = data[0]<mu[0]+sigmas[0] and data[0]>mu[0] - sigmas[0]
    ycheck = data[1]<mu[1]+sigmas[1] and data[1]>mu[1] - sigmas[1]
    
    return xcheck and ycheck


def is_inside_sigma_vector(data, mu, covariance):
    # data = np.array(       [ [x1,y1],        [x2,y2]       ,...])  <-- ground truth
    # mu = np.array(         [ [mux1,muy1],    [mux2,muy2]   ,...])  <-- from nn
    # covariance = np.array([[ [x,xy],[xy,y]], [x,xy],[xy,y]],...])  <-- from nn
    bool_vector= []
    counts = 0 
    for i in range(len(data)):
        bool_vector.append(is_inside_sigma(data[i], mu[i], covariance[i]))
        if bool_vector[-1]:
            counts+=1
    return bool_vector, counts
    

def plot_result(counts,totals,dt,label="",title=""):
    # counts = np.array([n1,n2,..])
    # totals = np.array([N1,N2,..]) maybe all N is equal
    # dt -> just for current visualisation
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    x = np.array([i*dt  for i in range(0,len(counts))])
    y = counts/totals
    ax.set_ylim(-0.1,1.1)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('hits_in_sigma to total')
    ax.plot(x, y, label=label)
    ax.set_title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show(True)

if __name__ == "__main__":

    # a = np.load("data.npz")
    # gt =   a["gt_"]
    # pred = a["pred"]
    # cov =  a["cov"]


    # checks, count = is_inside_sigma_vector(gt,pred,cov)

    data = np.array(      [[10,10],[20,20],[30,30]])
    mu = np.array(        [[11,5], [21,20], [31,31]])
    covariance = np.array([[[1.5,0.5],[0.5,1.5]],
                            [[1.5,0.5],[0.5,1.5]],
                            [[1.5,0.5],[0.5,1.5]]])

    mv = calculate_moholonobis_vector(data,mu,covariance)
    print(mv)

    checks, count = is_inside_sigma_vector(data,mu,covariance)
    print(checks)
    print(count)

    counts = np.array([800,600,500,450,300,290,285])
    totals = np.array([1000,1000,1000,1000,1000,1000,1000])


    plot_result(counts,totals, dt = 0.1)