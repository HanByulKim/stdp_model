####
# STDP pytorch model for multi patterns
#

from tensorflow.examples.tutorials.mnist import input_data
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
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

ts=10 # timestep
tt=len(input[:]) # step number

size = 28
N = 1000
M = 1   # Batch
train_size = 1000
Dis = 10  # Display Var

# Wmax Wmin
Wma = 260*(10**-4)
Wmax = torch.FloatTensor(N,size*size).uniform_(Wma,Wma).cuda()
Wmi = 56*(10**-8) #-56*(10**-3)
Wmin = torch.FloatTensor(N,size*size).uniform_(Wmi,Wmi).cuda()

# Init Weight
# W = np.random.uniform(low=Wmin, high=Wmax, size=(tt+2,size*size))
W = torch.FloatTensor(N,size*size).uniform_(Wmi,Wma).cuda()
classify = torch.FloatTensor(N).zero_().cuda()

# Threshold
Vth = 0.65
Vtheta = torch.FloatTensor(N).zero_().cuda()
Vtheta_unit = 0.1
Vdelta = math.exp(-1/12)

c = 60*(10**-1)
comp_z = torch.FloatTensor(1,size*size).zero_().cuda()

# LTP var
alpha_p = 0.01  #8.5*(10**-12)
beta_p = 2.35 #1.35

# LTD var
A=1.91*(10**-11)
B=-0.71865
C=7.44*(10**9)
D=-1.12*(10**19)

# Inhibitory Class 10
#InTen = torch.FloatTensor(N,10).uniform_(Wmi/3,Wma/3).cuda()

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

# Train
while(i < train_size):
    Qe = torch.FloatTensor(N).zero_().cuda()
    Vi = torch.FloatTensor(N).zero_().cuda()
    Ii = torch.FloatTensor(N).zero_().cuda()
    fire = torch.FloatTensor(N).zero_().cuda()
    print("---" + str(i))
    
    for j in range(0,60):
        # Y = W*X, Y as current to potential
        sampling = torch.bernoulli(input[:,i:i+M]).cuda()
        #sampling = sampling - (1 - sampling)
        Ie = torch.sum(W.mm(sampling), dim=1)
        #print(Ie[0])
        
        #Q = (1-fire) * (I*ts*(10**-3) + Q) + fire * (I*ts*(10**-3))
        # Exc Neuron
        Qe = Ie*ts*(10**-2) + Qe
        Ve = Qe/c
        # In Neuron
        Vi = Vi + ((-60*(10**-3) - Vi) + (Ie - Ii))
        Ii = c*Vi/ts

        #print(str(i) + " : " + str(Vth+Vtheta) + " / " + str(V))
        #file2.write(str(i) + " : " + str(Vth+Vtheta) + " / " + str(V) + "\n")
        # Vth exponentially decayed
        Vtheta *= Vdelta
        
        fire = Vi.gt(Vth + Vtheta).float().cuda()
        Vtheta = Vtheta.add(fire*Vtheta_unit)

        # clamping on Wmax Wmin
        #gt = Wmax.gt(W).float()
        #ge = W.ge(Wmin + (A+B*Wmin+C*(Wmin*Wmin)+D*(Wmin*Wmin*Wmin))).float()
        #ge = W.ge(Wmin).float()
	
        # init -> LTP, (1-init) -> LTD
        update = fire.expand(size*size,N).transpose(1,0).float().cuda()
        init = sampling.transpose(1,0).expand(N,size*size)
        # input > 0 LTP
        LTPmask = update * init
        # input < 0 LTP
        LTDmask = update * (1 - init)
        # Weight Update
        #dw =  LTPmask * (alpha_p*(-beta_p*W.sub(Wmin)/Wmax.sub(Wmin)).exp()) + LTDmask * (- (A+B*W+C*(W*W)+D*(W*W*W)))
        dw =  LTPmask * (alpha_p * (Wma-W)) + LTDmask * (-1 * alpha_p * (Wma-W) )
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