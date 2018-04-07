####
# STDP supervised model for multi patterns
#

from tensorflow.examples.tutorials.mnist import input_data
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
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
label = mnist.train.labels

tt=len(input[:]) # step number

size = 28
post_size = 30
N = post_size*10
M = 1   # Batch
train_size = 1000
Dis = 10  # Display Var

# Wmax Wmin
Wma = 0.02
Wmax = torch.FloatTensor(N,size*size).uniform_(Wma,Wma).cuda()
Wmi = 0 #-56*(10**-3)
Wmin = torch.FloatTensor(N,size*size).uniform_(Wmi,Wmi).cuda()

# Init Weight
# W = np.random.uniform(low=Wmin, high=Wmax, size=(tt+2,size*size))
W = torch.FloatTensor(N,size*size).uniform_(Wmi,Wma).cuda()

# Threshold
Vth = 0.65
Vtheta = torch.FloatTensor(N).zero_().cuda()
Vtheta_unit = 0.1
Vdelta = math.exp(-1/12)

c = 60*(10**-1)
comp_z = torch.FloatTensor(1,size*size).zero_().cuda()

# Trining var
alpha_p = 0.01  #8.5*(10**-12)
alpha_d = -0.01
beta_p = 2.35 #1.35

# Display Var
temp = torch.FloatTensor(size,size*Dis).zero_().cuda()
temp_in = torch.FloatTensor(size,size*Dis).zero_().cuda()
X = np.linspace(-25*Dis,25*Dis,size*Dis)
Y = np.linspace(-25,25,size)

# graph animation
plt.ion()
fig = plt.figure()
#fig2 = plt.figure()
ax = fig.add_subplot(111)
#ax2 = fig2.add_subplot(111)
ax.axis([-25*Dis,25*Dis,-25,25])
#ax2.axis([-25*N,25*N,-25,25])
plt.draw()

# First Pic
#for x in range(0,size):
#	temp[x] = W[0][x*size:(x+1)*size]

ax.pcolormesh(X,Y,temp,cmap=plt.cm.get_cmap('RdBu'))
plt.title('weight init')
plt.pause(0.00001)
i=0
inTarget = 20
deTarget = 0
train_step = 2
Tn = 6
ts = 5*(10**-3) # timestep 5ms

P = torch.FloatTensor(size*size).zero_().cuda()
Q = torch.FloatTensor(N).zero_().cuda()
preNeuron = torch.IntTensor(size*size, 2*Tn).zero_().cuda()
postNeuron = torch.IntTensor(N, 2*Tn).zero_().cuda()

ALTP = 1
ALTD = -1
tau_ltp = 4*ts
tau_ltd = 4*ts
aLTP = 6*(10**-5)
aLTD = 6.3*(10**-5)

def updateP(dt):
    return P * math.exp(dt/tau_ltp) + ALTP

def updateQ(dt):
    return Q * math.exp(dt/tau_ltd) + ALTD

#%%
# Train
while(i < train_size):
    # Artificial Spiking List
    fire = torch.FloatTensor(N).zero_().cuda()
    deList = torch.IntTensor(0).zero_().cuda()
    inList = torch.IntTensor(0).zero_().cuda()
    holdList = torch.IntTensor(0).zero_().cuda()
    
    # Recent Spike Timing
    print("---" + str(i))
    
    # Natural Output Spiking
    sampling = torch.bernoulli(input[:,i:i+M]).cuda()
    out = W.mm(sampling)
    fire = out.gt(Vth).float().cuda()
    
    # Superising Labelsssssssss
    id = np.argmax(mnist.train.labels[i], axis=0)
    
    # Making Artificial List    
    for j in range(0,10):
        numspike = torch.sum(fire[j*post_size:(j+1)*post_size],dim=0).int()[0]
        if(j == id):
            #add all spiking neurons to holdlist
            inList = torch.cat((inList, torch.IntTensor(post_size).zero_().cuda()), 0)
            holdList = torch.cat((holdList, fire[j*post_size:(j+1)*post_size,0].int()), 0)
            
            if(numspike < inTarget):
                # add x non-spiking neurons to inlist
                x = inTarget - numspike
                cand = ((1-fire[j*post_size:(j+1)*post_size]) * torch.FloatTensor(post_size,1).uniform_(0,1).cuda())[:,0]
                a, ade = cand.topk(x)
                
                deList = torch.cat((deList, cand.ge(a[x-1]).int()),0)
            else:
                deList = torch.cat((deList, torch.IntTensor(post_size).zero_().cuda()),0)
        elif(numspike > deTarget):
            #add y spiking neurons to delist
            y = numspike - deTarget
            cand = (fire[j*post_size:(j+1)*post_size] * torch.FloatTensor(post_size,1).uniform_(0,1).cuda())[:,0]
            a, ain = cand.topk(y)
            
            inList = torch.cat((inList, cand.ge(a[y-1]).int()),0)
            deList = torch.cat((deList, torch.IntTensor(post_size).zero_().cuda()),0)
            holdList = torch.cat((holdList, torch.IntTensor(post_size).zero_().cuda()),0)
        else:
            inList = torch.cat((inList, torch.IntTensor(post_size).zero_().cuda()),0)
            deList = torch.cat((deList, torch.IntTensor(post_size).zero_().cuda()),0)
            holdList = torch.cat((holdList, torch.IntTensor(post_size).zero_().cuda()),0)
    i+=1
      #%%                

    # Training on Trainstep
    for j in range(0, train_step):
        # new training phase : clear all
        tpreRecent = torch.IntTensor(size*size).uniform_(-math.inf,-math.inf).cuda()
        tpostRecent = torch.IntTensor(N).uniform_(-math.inf,-math.inf).cuda()
        
        for k in range(0, Tn):
            if(k==1):  # deList
                postNeuron[:,k] = deList
                dpost = (tpostRecent - deList * k) * ts
                tpostRecent = (1-deList) * tpostRecent + deList * (j*train_step + k)
                Q = updateQ(dpost)
                
            if(k==2):  # input
                preNeuron[:,k] = sampling[:,0]
                dpre = (tpreRecent - sampling[:,0] * k) * ts
                tpreRecent = (1-sampling[:,0]) * tpreRecent + sampling[:,0] * (j*train_step + k)
                P = updateP(dpre)
                
                #LTD
                #LTD from hold
                
            if(k==3):  # inList
                postNeuron[:,k] = inList
                dpost = (tpostRecent - inList * k) * ts
                tpostRecent = (1-inList) * tpostRecent + inList * (j*train_step + k)
                Q = updateQ(dpost)
                
                #LTP
                
            if(k==4):
                #LTP from natural increase
                
            if(j!=0 and k==0):  # holdList
                postNeuron[:,k] = holdList
                dpost = (j*Tn + k) - dpost
                
            # update equation            
            dWq = aLTD * Q * 
            dWp = aLTP * P *
            
            # Y = W*X, Y as current to potential
            sampling = torch.bernoulli(input[:,i:i+M]).cuda()
            I = torch.sum(W.mm(sampling), dim=1)
            
            Q = (1-fire) * (I*ts*(10**-3) + Q) + fire * (I*ts*(10**-3))
            V = Q/c
    
            #print(str(i) + " : " + str(Vth+Vtheta) + " / " + str(V))
            #file2.write(str(i) + " : " + str(Vth+Vtheta) + " / " + str(V) + "\n")
            # Vth exponentially decayed
            Vtheta *= Vdelta
            
            fire = Vi.gt(Vth + Vtheta).float().cuda()
            Vtheta = Vtheta.add(fire*Vtheta_unit)
    	
            # init -> LTP, (1-init) -> LTD
            update = fire.expand(size*size,N).transpose(1,0).float().cuda()
            init = sampling.transpose(1,0).expand(N,size*size)
            # input > 0 LTP
            LTPmask = update * init
            # input < 0 LTP
            LTDmask = update * (1 - init)
            # Weight Update
            #dw =  LTPmask * (alpha_p*(-beta_p*W.sub(Wmin)/Wmax.sub(Wmin)).exp()) + LTDmask * (- (A+B*W+C*(W*W)+D*(W*W*W)))
            dw =  LTPmask * (alpha_p * Q * (Wma-W)) + LTDmask * (alpha_d * (Wma-W) )
            W.add_(dw)
            
            #W = update * ( init * (gt * (W + (alpha_p*(-beta_p*W.sub(Wmin)/Wmax.sub(Wmin)).exp())) + (1-gt) * Wmax) + (1 - init) * ( ge * (W - (A+B*W+C*(W*W)+D*(W*W*W))) + (1-ge) * Wmin ) ) + (1 - update) * W
    
            # clamping
            W.clamp_(Wmi,Wma)
        
            #print(str(i) + " : " + str(fire[0:10]))
            #file.write(str(i) + " : " + str(fire) + "\n")
            #print(str(i) + "-" + str(j) + "Vth : " + str(Vth + Vtheta))
            #print(str(i) + "-" + str(j) + "Vi : " + str(Vi.cpu().numpy()))
            #file2.write(str(i) + "-" + str(j) + "Vth : " + str(Vth + Vtheta) + "\n")
            #file3.write(str(i) + "-" + str(j) + "V : " + str(Vi.cpu().numpy()) + "\n")
            #print("Weight" + str(i) + "-" + str(j) + " : " + str(W))
            #file3.write(str(i) + "-" + str(j) + " : " + str(W.cpu().numpy()) + "\n")
    
            # Displaying	
            if(i==train_size-1 and j ==29):
                for h in range(0,Dis): #N
                    for k in range (0,size):
                        temp[k][h*size:h*size+size] = W[h][k*size:k*size+size]
                        temp_in[k][h*size:h*size+size] = input[k*size:k*size+size, i + h]
    					  
                #ax2.pcolormesh(X,Y,temp_in,cmap=plt.cm.get_cmap('gray'))
                ax.pcolormesh(X,Y,temp,cmap=plt.cm.get_cmap('RdBu'))
                plt.title('weight ' + str(i) + '-' + str(j))
                #plt.savefig('log/' + str(i) + '-' + str(j) + '.png')
                plt.savefig('vex.png')
                #plt.pause(0.00000000001)

    P = P * (train_step/Tltp) + Altp
    Q = Q * (train_step/Tltd) + Altd
    dwp = altd*Q*
    i+=M

#%%
# classify
# << need to add avg concept
result = torch.FloatTensor(N,10).zero_().cuda()
label = np.zeros((N))
for i in range(train_size, len(input[0,:])):
    result += W.mm(input[:,i:i+1]).expand(N,10) * torch.from_numpy(mnist.train.labels[i]).expand(N,10).float().cuda()

for i in range(0, len(result[:])):
    a, b = result[i].max(0)
    label[i] = b[0]

print(label)

#%%
# Test
i=0
while(i < len(test[:])):
    Y = W.mm(test[:,0].expand(1,size*size).transpose(1,0))
    Res = torch.max(Y)
    i += 1

file.close()
file2.close()
file3.close()