from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from numba import autojit

# Log
file = open("log/fire.txt","w")
file2 = open("log/Volt.txt","w")

# Data Read
# mnist.train.images => [55000,784](28*28=784)
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)
input = mnist.train.images
label = mnist.train.labels
train_pattern = 1

ts=10
tt=len(input[:]) # step number

size = 28
Wmax = 260*(10**-12)
Wmin = 56*(10**-12)
Vth = 400
c = 500*(10**-15)

# Init Weight
W = np.random.uniform(low=Wmin, high=Wmax, size=(tt+2,size*size))

I = np.zeros((tt+1,size*size))
fire = np.zeros((tt+1,1))

# LTP
alpha_p=8.5*(10**-12)
beta_p=1.35

#LTD
A=1.91*(10**-11)
B=-0.71865
C=7.44*(10**9)
D=-1.12*(10**19)

V = np.zeros((tt+1,1))
Q = np.zeros((tt+1,1))
temp = np.zeros((size,size))
temp_in = np.zeros((size,size))
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
			I[i] = input[cnt][:]*W[i][:]

			if(i==0):
				Q[i] = np.sum(I[i])*ts*(10**-3)
				V[i] = Q[i]/c
			else:
				Q[i] = np.sum(I[i])*ts*(10**-3)+Q[i-1]
				V[i] = Q[i]/c
		else:
			#I[i] = input[cnt][:] * W[i][:]
			I[i] = input[cnt][:]*W[i][:]
			Q[i] = np.sum(I[i])*ts*(10**-3)
			V[i] = Q[i]/c

		W[i+1]=W[i]
		print(str(i)+" : "+str(V[i]))
		file2.write(str(i)+" : "+str(V[i])+"\n")

		if(V[i]>Vth):
			fire[i+1]=1
			V[i+1]=0
			Q[i+1]=0
			for j in range(0,size*size):
				if(input[cnt][j] > 0):
					if(W[i][j] < Wmax):
						W[i+1][j] = W[i][j] + alpha_p*math.exp(-beta_p*(W[i][j]-Wmin)/(Wmax-Wmin))
					else:
						W[i+1][j] = Wmax
				else:
					if(W[i][j] >= Wmin + (A+B*Wmin+C*(Wmin**2)+D*(Wmin**3))):
						W[i+1][j] = W[i][j] - (A+B*W[i][j]+C*(W[i][j]**2)+D*(W[i][j]**3))
					else:
						W[i+1][j] = Wmin

		# Displaying
		for x in range(0,size):
			for y in range(0,size):
				temp[x][y] = W[i][x*size + y]
				temp_in[x][y] = input[cnt][x*size + y]
				
		
		ax2.pcolormesh(X,Y,temp_in,cmap=plt.cm.get_cmap('gray'))
		ax.pcolormesh(X,Y,temp,cmap=plt.cm.get_cmap('RdBu'))
		plt.title('weight '+str(i))
		plt.pause(0.0000000001)
		print(str(i)+" : "+str(fire[i]))
		file.write(str(i)+" : "+str(fire[i])+"\n")
		i+=1

file.close()
file2.close()
