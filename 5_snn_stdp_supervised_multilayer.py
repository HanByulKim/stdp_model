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

i = 0
acc = 0

# train step
train_step = 2
Tn = 6
ts = 5*(10**-3) # timestep 5ms
train_size = len(input[0,:])
test_size = len(test[0,:])
size = 28
N = 300
inf = -10**8

Wmax = 0.02
Wmin = 0
Vth = 0.6
ALTP = 1
ALTD = -1
tau_ltp = 4*ts
tau_ltd = 4*ts
aLTP = 6*(10**-5)
aLTD = 6.3*(10**-5)

inTarget = 20
deTarget = 0

class NN:
    def __init__(self, size, N, M, Wmax, Wmin, Vth, ALTP, ALTD, tau_ltp, tau_ltd, aLTP, aLTD, inTarget, deTarget):
        self.size = size
        self.N = N
        self.post_size = int(self.N/10)
        self.M = M   # Batch
        
        # Wmax Wmin
        self.Wmax = Wmax
        self.Wmin = Wmin #-56*(10**-3)
        
        # Init Weight
        self.W = torch.FloatTensor(self.N,self.size*self.size).uniform_(self.Wmin,self.Wmax).cuda()
        
        # Threshold
        self.Vth = Vth #0.6
        
        # Training Var
        self.ALTP = ALTP
        self.ALTD = ALTD
        self.tau_ltp = tau_ltp
        self.tau_ltd = tau_ltd
        self.aLTP = aLTP
        self.aLTD = aLTD
        
        # Targeted number of spiking
        self.inTarget = inTarget
        self.deTarget = deTarget
        
        # Neuron Coefficient
        self.P = torch.FloatTensor(self.size*self.size).zero_().cuda()
        self.Q = torch.FloatTensor(self.N).zero_().cuda()

    # updateing LTP coefficient P
    def updateP(self, dt, spike):
        self.P = spike * (self.P * (dt/self.tau_ltp).exp() + self.ALTP) + (1-spike) * self.P
    
    # updateing LTD coefficient Q
    def updateQ(self, dt, spike):
        self.Q = spike * (self.Q * (dt/self.tau_ltd).exp() + self.ALTD) + (1-spike) * self.Q
    
    # Artificial decrease
    def adec(self, sample, coeff, dt, List): # tpost < tpre (deList, holdList)
        # decrease filter = decrease list * input spike == 1
        filt = List.expand(self.size*self.size,self.N).transpose(1,0).cuda() * sample.expand(self.size*self.size,self.N).transpose(1,0).cuda()
        
        return filt * self.aLTD * self.Q.expand(self.size*self.size,self.N).transpose(1,0) * math.exp(dt/self.tau_ltd)
    
    # Artificial increase
    def ainc(self, sample, coeff, dt, List): # tpre < tpost (inList)
        # increase filter = input spike == 1 * increase list
        filt = sample.expand(self.size*self.size,self.N).transpose(1,0).cuda() * List.expand(self.size*self.size,self.N).transpose(1,0).cuda()
        
        return filt * self.aLTP * self.P.expand(self.N,self.size*self.size) * math.exp(dt/self.tau_ltp)
    
    # Natural Increase
    def ninc(self, sample, coeff, dt, invList): # natural increase
        # natural inc. filer = input spike == 1 * no de, in ,hold
        filt = sample.expand(self.size*self.size,self.N).transpose(1,0).cuda() * (1 - invList).expand(self.size*self.size,self.N).transpose(1,0).cuda()
        
        return filt * self.aLTP * self.P.expand(self.N,self.size*self.size) * math.exp(dt/self.tau_ltp)
    
    # multiplication with W
    def Wmul(self, sample):
        return self.W.mm(sample)
    
    # add dw
    def Wadd(self, dw):
        self.W.add_(dw)
    
    def Wclamp(self):
        self.W.clamp_(self.Wmin,self.Wmax)
        
class List:
    def __init__(self):
        self.post_size = 30
        self.N = self.post_size*10
        self.deList = torch.LongTensor(0).zero_().cuda() # Artificial decreasing List
        self.inList = torch.LongTensor(0).zero_().cuda() # Artificial increasing List
        self.holdList = torch.LongTensor(0).zero_().cuda() # Holding list
        
    def ZEROdeList(self):
        self.deList = torch.cat((self.deList, torch.LongTensor(self.post_size).zero_().cuda()), 0)
        
    def ZEROinList(self):
        self.inList = torch.cat((self.inList, torch.LongTensor(self.post_size).zero_().cuda()),0)
        
    def ZEROholdList(self):
        self.holdList = torch.cat((self.holdList, torch.LongTensor(self.post_size).zero_().cuda()),0)
        
    def ADDholdList(self, spike):
        self.holdList = torch.cat((self.holdList, spike.long()), 0)
        
    def ADDinList(self, spike, num):
        cand = ((1-spike) * torch.FloatTensor(self.post_size,1).uniform_(0,1).cuda())[:,0]
        self.inList = torch.cat((self.inList, cand.ge(cand.topk(num)[0][x-1]).long()),0)
        
    def ADDdeList(self, spike, num):
        cand = (spike * torch.FloatTensor(self.post_size,1).uniform_(0,1).cuda())[:,0]        
        self.deList = torch.cat((self.deList, cand.ge(cand.topk(y)[0][y-1]).long()),0)

layer1 = NN(size, 300, 1, Wmax, Wmin, Vth, ALTP, ALTD, tau_ltp, tau_ltd, aLTP, aLTD, inTarget, deTarget)

# Train
i=0
while(i < train_size):
    list1 = List()
    # Artificial Spiking List
    fire = torch.FloatTensor(N).zero_().cuda()
    
    # Step
    if(i%100 == 0):
        print("---" + str(i))
    
    # Natural Output Spiking
    sampling = torch.bernoulli(input[:,i:i+layer1.M]).cuda()
    out = layer1.Wmul(sampling)
    file2.write(str(out))
    fire = out.gt(layer1.Vth).float().cuda()
    
    # Superising Labels
    id = np.argmax(mnist.train.labels[i], axis=0)
    
    # Making Artificial List    
    for j in range(0,10):
        numspike = torch.sum(fire[j*layer1.post_size:(j+1)*layer1.post_size],dim=0).int()[0]
        if(j == id):
            #add all spiking neurons to holdlist
            list1.ZEROdeList()
            list1.ADDholdList(fire[j*list1.post_size:(j+1)*list1.post_size,0])
            
            if(numspike < layer1.inTarget):
                # add x non-spiking neurons to inlist
                x = layer1.inTarget - numspike
                list1.ADDinList(fire[j*list1.post_size:(j+1)*list1.post_size], x)
            else:
                list1.ZEROinList()
        elif(numspike > layer1.deTarget):
            #add y spiking neurons to delist
            y = numspike - layer1.deTarget
            list1.ADDdeList(fire[j*list1.post_size:(j+1)*list1.post_size], y)
            list1.ZEROinList()
            list1.ZEROholdList()
        else:
            list1.ZEROinList()
            list1.ZEROdeList()
            list1.ZEROholdList()

    # Training on Trainstep
    for j in range(0, train_step):
        # new training phase : clear all
        tpreRecent = torch.LongTensor(layer1.size*layer1.size).zero_().cuda()
        tpreRecent += inf
        tpostRecent = torch.LongTensor(layer1.N).zero_().cuda()
        tpostRecent += inf
        for k in range(0, Tn):
            # T1
            if(k==1):  # deList
                # Q update
                dpost = (tpostRecent - list1.deList * (j*Tn + k)).float() * ts
                tpostRecent = (1-list1.deList) * tpostRecent + list1.deList * (j*Tn + k)
                layer1.updateQ(dpost, list1.deList.float())
                
            # T2
            if(k==2):  # input
                # P update
                dpre = (tpreRecent.float() - sampling[:,0] * (j*Tn + k)) * ts
                tpreRecent = (1-sampling[:,0].long()) * tpreRecent + sampling[:,0].long() * (j*Tn + k)
                layer1.updateP(dpre, sampling[:,0].float())
                
                # LTD from artificial decrease
                dw = layer1.adec(sampling, 0, -ts, list1.deList.float())
                layer1.Wadd(dw)
                # LTD from hold
                if(j!=0):
                    dw = layer1.adec(sampling, 0, -2*ts, list1.holdList.float())
                    layer1.Wadd(dw)
                
            # T3
            if(k==3):  # inList
                # Q update
                dpost = (tpostRecent - list1.inList * (j*Tn + k)).float() * ts
                tpostRecent = (1-list1.inList) * tpostRecent + list1.inList * (j*Tn + k)
                layer1.updateQ(dpost, list1.inList.float())
                
                # LTP from artificial increase
                dw = layer1.ainc(sampling, 0, -ts, list1.inList.float())
                layer1.Wadd(dw)
                
            # T4
            if(k==4):
                # LTP from natural increase
                if(j==0):
                    dw = layer1.ninc(sampling, 0, -2*ts, (list1.inList + list1.deList).float())
                    layer1.Wadd(dw)
                else:
                    dw = layer1.ninc(sampling, 0, -2*ts, (list1.inList + list1.deList + list1.holdList).float())
                    layer1.Wadd(dw)
                
            # Tn + T0
            if(j!=0 and k==0):  # holdList
                # Q update
                dpost = (tpostRecent - list1.holdList * (j*Tn + k)).float() * ts
                tpostRecent = (1-list1.holdList) * tpostRecent + list1.holdList * (j*Tn + k)
                layer1.updateQ(dpost, list1.holdList.float())
            
            # update W
            layer1.Wclamp()

            #print(str(i) + " : " + str(fire[0:10]))
            #file.write(str(i) + " : " + str(fire) + "\n")
            #print(str(i) + "-" + str(j) + "Vth : " + str(Vth + Vtheta))
            #print(str(i) + "-" + str(j) + "Vi : " + str(Vi.cpu().numpy()))
            #file2.write(str(i) + "-" + str(j) + "Vth : " + str(Vth + Vtheta) + "\n")
            #file3.write(str(i) + "-" + str(j) + "V : " + str(Vi.cpu().numpy()) + "\n")
            #print("Weight" + str(i) + "-" + str(j) + " : " + str(W))
            #file3.write(str(i) + "-" + str(j) + " : " + str(W.cpu().numpy()) + "\n")
    i+=layer1.M
#%%
# Test
fire = torch.FloatTensor(layer1.N).zero_().cuda()
result = torch.IntTensor(10).zero_().cuda()
acc=0
for i in range(0, test_size):
    out = layer1.Wmul(test[:,i:i+layer1.M])
    fire = out.gt(out.mean(dim=0) + 1.6*out.std(dim=0)).float().cuda()
    
    for j in range(0,10):
        result[j] = torch.sum(fire[j*layer1.post_size:(j+1)*layer1.post_size],dim=0).int()[0]
    
    # Superising Labels
    if(torch.max(result, dim=0)[1][0] == np.argmax(mnist.test.labels[i], axis=0)):
        acc += 1
        print("match " + str(torch.max(result, dim=0)[1][0]) + " == " + str(np.argmax(mnist.test.labels[i], axis=0)))
    else:
        print("wrong " + str(torch.max(result, dim=0)[1][0]) + " == " + str(np.argmax(mnist.test.labels[i], axis=0)))

print("accuracy : " + str(acc))

file.close()
file2.close()
file3.close()