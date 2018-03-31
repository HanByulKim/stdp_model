####
# STDP pytorch model for only 1 pattern
#

from tensorflow.examples.tutorials.mnist import input_data
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# Log
file = open("log/fire.txt","w")
file2 = open("log/Volt.txt","w")
file3 = open("log/tempw.txt","w")

# Data Read
# mnist.train.images => [55000,784](28*28=784)
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
input = torch.from_numpy(mnist.train.images).cuda()
label = mnist.train.labels
train_pattern = 1

ts=10
tt=len(input[:]) # step number

size = 28
Wma = 260*(10**-12)
Wmax = torch.FloatTensor(1,size*size).uniform_(Wma,Wma).cuda()
Wmi = 56*(10**-12)
Wmin = torch.FloatTensor(1,size*size).uniform_(Wmi,Wmi).cuda()
Vth = 400
c = 500*(10**-15)

# Init Weight
#W = np.random.uniform(low=Wmin, high=Wmax, size=(tt+2,size*size))
W = torch.FloatTensor(tt+2,size*size).uniform_(Wmi,Wma).cuda()
comp_z = torch.FloatTensor(1,size*size).uniform_(0,0).cuda()

#I = np.zeros((tt+1,size*size))
#I = torch.FloatTensor(tt+1,size*size).zero_().cuda()
fire = np.zeros((tt+1,1))
#fire = torch.FloatTensor(tt+1,1).zero_().cuda()

# LTP
alpha_p=8.5*(10**-12)
beta_p=1.35

#LTD
A=1.91*(10**-11)
B=-0.71865
C=7.44*(10**9)
D=-1.12*(10**19)

V = np.zeros((tt+1,1))
#V = torch.FloatTensor(tt+1,1).zero_().cuda()
Q = np.zeros((tt+1,1))
#Q = torch.FloatTensor(tt+1,1).zero_().cuda()
#temp = np.zeros((size,size))
temp = torch.FloatTensor(size,size).zero_().cuda()
#temp_in = np.zeros((size,size))
temp_in = torch.FloatTensor(size,size).zero_().cuda()
X = np.linspace(-25,25,size)
Y = np.linspace(-25,25,size)

# graph animation
plt.ion()
fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax.axis([-25,25,-25,25])
ax2.axis([-25,25,-25,25])
plt.draw()

# First Pic
for x in range(0,size):
	for y in range(0,size):
		temp[x][y] = W[0][x*size + y]

ax.pcolormesh(X,Y,temp,cmap=plt.cm.get_cmap('RdBu'))
plt.title('weight init')
plt.pause(0.00001)

# Train
i=0
for cnt in range(0,len(input[:])):
	if(label[cnt][train_pattern]==1):
		if(fire[i]==0):
			#I[i] = input[cnt][:]*W[i][:]
			I = input[cnt].dot(W[i])

			if(i==0):
				Q[i] = I*ts*(10**-3)
				V[i] = Q[i]/c
			else:
				Q[i] = I*ts*(10**-3)+Q[i-1]
				V[i] = Q[i]/c
		else:
			#I[i] = input[cnt][:] * W[i][:]
			I = input[cnt].dot(W[i])
			Q[i] = I*ts*(10**-3)
			V[i] = Q[i]/c

		W[i+1]=W[i]
		print(str(i)+" : "+str(V[i]))
		file2.write(str(i)+" : "+str(V[i])+"\n")
        
		if(V[i]>Vth):	# updates triggered only when a spike is fired by a postsynaptic excitatory neuron
			fire[i+1]=1
			V[i+1]=0
			Q[i+1]=0
			#for j in range(0,size*size):
				#if(input[cnt][j] > 0):
					#if(W[i][j] < Wmax):
					#W[i+1][j] = W[i][j] + alpha_p*math.exp(-beta_p*(W[i][j]-Wmin)/(Wmax-Wmin))
			init = input[cnt].gt(comp_z).float()
			gt = Wmax.gt(W[i]).float()
			ge = W[i].ge(Wmin + (A+B*Wmin+C*(Wmin*Wmin)+D*(Wmin*Wmin*Wmin))).float()
			W[i+1] = init * (gt * (W[i] + (alpha_p*(-beta_p*W[i].sub(Wmin)/Wmax.sub(Wmin)).exp())) + (1-gt) * Wmax) + (1 - init) * ( ge * (W[i] - (A+B*W[i]+C*(W[i]*W[i])+D*(W[i]*W[i]*W[i]))) + (1-ge) * Wmin )
			#W[i+1] = W[i].sub(Wmin)
				#else:
					#if(W[i][j] >= Wmin + (A+B*Wmin+C*(Wmin**2)+D*(Wmin**3))):

			#file3.write("W[i+1] torch " + str(i))
			#file3.write(str(gt*(W[i].sub(Wmin))))
			#file3.write("\n W[i+1] ori " + str(i))

		# Displaying
		for x in range(0,size):
			for y in range(0,size):
				temp[x][y] = W[i][x*size + y]
				temp_in[x][y] = input[cnt][x*size + y]
				
		
		ax2.pcolormesh(X,Y,temp_in,cmap=plt.cm.get_cmap('gray'))
		ax.pcolormesh(X,Y,temp,cmap=plt.cm.get_cmap('RdBu'))
		plt.title('weight '+str(i))
		#plt.pause(0.00000000001)
		print(str(i)+" : "+str(fire[i]))
		file.write(str(i)+" : "+str(fire[i])+"\n")
		i+=1

file.close()
file2.close()
file3.close()