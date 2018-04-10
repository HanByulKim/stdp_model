####
# STDP supervised model for multi patterns
#

from tensorflow.examples.tutorials.mnist import input_data
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
#import HHneuron
np.set_printoptions(threshold=np.nan)
        
# Log
file = open("log/fire.txt","w")
file2 = open("log/Volt.txt","w")
file3 = open("log/Weight.txt","w")

# Data Read
# mnist.train.images => [55000,784](28*28=784)
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
input = torch.from_numpy(mnist.train.images.transpose(1,0)).cuda()
test = torch.from_numpy(mnist.test.images.transpose(1,0)).cuda()

size = 28
post_size = 30
N = post_size*10
M = 1   # Batch
train_size = len(input[0,:])
test_size = len(test[0,:])
acc = 0

# Wmax Wmin
Wmax = 0.02
Wmin = 0 #-56*(10**-3)

# Init Weight
W = torch.FloatTensor(N,size*size).uniform_(Wmin,Wmax).cuda()

# Threshold
Vth = 0.81

# train step
train_step = 2
Tn = 6
ts = 5*(10**-3) # timestep 5ms

# Training Var
i=0
ALTP = 1
ALTD = -1
tau_ltp = 4*ts
tau_ltd = 4*ts
aLTP = 6*(10**-5)
aLTD = 6.3*(10**-5)
inf = -10**8

# Targeted number of spiking
inTarget = 20
deTarget = 0

# Neuron Coefficient
P = torch.FloatTensor(size*size).zero_().cuda()
Q = torch.FloatTensor(N).zero_().cuda()

# Neuron spiking status
preNeuron = torch.LongTensor(size*size, 2*Tn).zero_().cuda()
postNeuron = torch.LongTensor(N, 2*Tn).zero_().cuda()

# updateing LTP coefficient P
def updateP(dt, spike):
    return spike * (P * (dt/tau_ltp).exp() + ALTP) + (1-spike) * P

# updateing LTD coefficient Q
def updateQ(dt, spike):
    return spike * (Q * (dt/tau_ltd).exp() + ALTD) + (1-spike) * Q

# Artificial decrease
def adec(coeff, dt, List): # tpost < tpre (deList, holdList)
    # decrease filter = decrease list * input spike == 1
    filt = List.expand(size*size,N).transpose(1,0).cuda() * sampling.expand(size*size,N).transpose(1,0).cuda()
    
    return filt * aLTD * Q.expand(size*size,N).transpose(1,0) * math.exp(dt/tau_ltd)

# Artificial increase
def ainc(coeff, dt, List): # tpre < tpost (inList)
    # increase filter = input spike == 1 * increase list
    filt = sampling.expand(size*size,N).transpose(1,0).cuda() * List.expand(size*size,N).transpose(1,0).cuda()
    
    return filt * aLTP * P.expand(N,size*size) * math.exp(dt/tau_ltp)

# Natural Increase
def ninc(coeff, dt, invList): # natural increase
    # natural inc. filer = input spike == 1 * no de, in ,hold
    filt = sampling.expand(size*size,N).transpose(1,0).cuda() * (1 - invList).expand(size*size,N).transpose(1,0).cuda()
    
    return filt * aLTP * P.expand(N,size*size) * math.exp(dt/tau_ltp)

#%%
# Train
while(i < train_size):
    # Artificial Spiking List
    fire = torch.FloatTensor(N).zero_().cuda()
    deList = torch.LongTensor(0).zero_().cuda() # Artificial decreasing List
    inList = torch.LongTensor(0).zero_().cuda() # Artificial increasing List
    holdList = torch.LongTensor(0).zero_().cuda() # Holding list
    
    # Recent Spike Timing
    print("---" + str(i))
    
    # Natural Output Spiking
    sampling = torch.bernoulli(input[:,i:i+M]).cuda()
    out = W.mm(sampling)
    file2.write(str(out))
    fire = out.gt(Vth).float().cuda()
    
    # Superising Labels
    id = np.argmax(mnist.train.labels[i], axis=0)
    
    # Making Artificial List    
    for j in range(0,10):
        numspike = torch.sum(fire[j*post_size:(j+1)*post_size],dim=0).int()[0]
        if(j == id):
            #add all spiking neurons to holdlist
            deList = torch.cat((deList, torch.LongTensor(post_size).zero_().cuda()), 0)
            holdList = torch.cat((holdList, fire[j*post_size:(j+1)*post_size,0].long()), 0)
            
            if(numspike < inTarget):
                # add x non-spiking neurons to inlist
                x = inTarget - numspike
                cand = ((1-fire[j*post_size:(j+1)*post_size]) * torch.FloatTensor(post_size,1).uniform_(0,1).cuda())[:,0]
                a, ade = cand.topk(x)
                
                inList = torch.cat((inList, cand.ge(a[x-1]).long()),0)
            else:
                inList = torch.cat((inList, torch.LongTensor(post_size).zero_().cuda()),0)
        elif(numspike > deTarget):
            #add y spiking neurons to delist
            y = numspike - deTarget
            cand = (fire[j*post_size:(j+1)*post_size] * torch.FloatTensor(post_size,1).uniform_(0,1).cuda())[:,0]
            a, ain = cand.topk(y)
            
            deList = torch.cat((deList, cand.ge(a[y-1]).long()),0)
            inList = torch.cat((inList, torch.LongTensor(post_size).zero_().cuda()),0)
            holdList = torch.cat((holdList, torch.LongTensor(post_size).zero_().cuda()),0)
        else:
            inList = torch.cat((inList, torch.LongTensor(post_size).zero_().cuda()),0)
            deList = torch.cat((deList, torch.LongTensor(post_size).zero_().cuda()),0)
            holdList = torch.cat((holdList, torch.LongTensor(post_size).zero_().cuda()),0)    

    # Training on Trainstep
    for j in range(0, train_step):
        # new training phase : clear all
        tpreRecent = torch.LongTensor(size*size).zero_().cuda()
        tpreRecent += inf
        tpostRecent = torch.LongTensor(N).zero_().cuda()
        tpostRecent += inf
        for k in range(0, Tn):
            # T1
            if(k==1):  # deList
                postNeuron[:,k] = deList
                
                # Q update
                dpost = (tpostRecent - deList * k).float() * ts
                tpostRecent = (1-deList) * tpostRecent + deList * (j*Tn + k)
                Q = updateQ(dpost, postNeuron[:,k].float())
                
            # T2
            if(k==2):  # input
                preNeuron[:,k] = sampling[:,0]
                
                # P update
                dpre = (tpreRecent.float() - sampling[:,0] * k) * ts
                tpreRecent = (1-sampling[:,0].long()) * tpreRecent + sampling[:,0].long() * (j*Tn + k)
                P = updateP(dpre, preNeuron[:,k].float())
                
                # LTD from artificial decrease
                dw = adec(0, -ts, deList.float())
                W.add_(dw)
                # LTD from hold
                if(j!=0):
                    dw = adec(0, -2*ts, holdList.float())
                    W.add_(dw)
                
            # T3
            if(k==3):  # inList
                postNeuron[:,k] = inList
                
                # Q update
                dpost = (tpostRecent - inList * k).float() * ts
                tpostRecent = (1-inList) * tpostRecent + inList * (j*Tn + k)
                Q = updateQ(dpost, postNeuron[:,k].float())
                
                # LTP from artificial increase
                dw = ainc(0, -ts, inList.float())
                W.add_(dw)
                
            # T4
            if(k==4):
                # LTP from natural increase
                if(j==0):
                    dw = ninc(0, -2*ts, (inList + deList).float())
                    W.add_(dw)
                else:
                    dw = ninc(0, -2*ts, (inList + deList * holdList).float())
                    W.add_(dw)
                
            # Tn + T0
            if(j!=0 and k==0):  # holdList
                postNeuron[:,k] = holdList
                
                # Q update
                dpost = (tpostRecent - holdList * k).float() * ts
                tpostRecent = (1-holdList) * tpostRecent + holdList * (j*Tn + k)
                Q = updateQ(dpost, postNeuron[:,k].float())
            
            # update W
            W.clamp_(Wmin,Wmax)

            #print(str(i) + " : " + str(fire[0:10]))
            #file.write(str(i) + " : " + str(fire) + "\n")
            #print(str(i) + "-" + str(j) + "Vth : " + str(Vth + Vtheta))
            #print(str(i) + "-" + str(j) + "Vi : " + str(Vi.cpu().numpy()))
            #file2.write(str(i) + "-" + str(j) + "Vth : " + str(Vth + Vtheta) + "\n")
            #file3.write(str(i) + "-" + str(j) + "V : " + str(Vi.cpu().numpy()) + "\n")
            #print("Weight" + str(i) + "-" + str(j) + " : " + str(W))
            #file3.write(str(i) + "-" + str(j) + " : " + str(W.cpu().numpy()) + "\n")
    i+=M

#%%
# Test
for i in range(0,test_size):
    sampling = torch.bernoulli(test[:,i:i+M]).cuda()
    out = W.mm(sampling)
    fire = out.gt(Vth).float().cuda()
    
    result = torch.IntTensor(10).cuda()
    
    for j in range(0,10):
        result[j] = torch.sum(fire[j*post_size:(j+1)*post_size],dim=0).int()[0]
    
    # Superising Labels
    if(torch.max(result, dim=0)[1][0] == np.argmax(mnist.test.labels[i], axis=0)):
        acc += 1
        print("match")
    else:
        print("wrong")

print("accuracy : " + str(acc))

file.close()
file2.close()
file3.close()