import numpy as np
import random as rand
import matplotlib.pyplot as plt
from random import random,seed
import math

def sphere(x):
  d=x.shape[0]
  sum=0
  for i in range(d):
    sum=sum+x[i]**2
  
  return sum

def rastrigin(x):
  d=x.shape[0]
  sum=0
  for i in range(d):
    sum=sum+x[i]**2-10*np.cos(2*np.pi*x[i])
  
  return 10*d+sum



num_variables=2
generation=0
x_max=5.12
x_min=-5.12


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.grid(True)


# create data points
xx = np.linspace(-5.12, 5.12, 100)
yy = np.linspace(-5.12, 5.12, 100)
# create grid
x1, x2 = np.meshgrid(xx, yy)

l=[x1,x2]
arr = np.array(l)

z = rastrigin(arr)

#fig, ax = plt.subplots()
# set labels


c1=2.05
c2=2.05
ini_v=3

maxiter = 100
w0 = 0.9
wf=0.1
slope = (wf-w0)/maxiter
w= w0
max_v = ini_v/3


S=20 #número de partículas

d=2

x=np.zeros((S,d))
v=np.zeros((S,d))
y=np.zeros((S,d))
    
part,dim=x.shape
    
    
f_ind=1e10*np.ones([part,1])
fx=np.zeros((part,1))
    
for i in range(part):
    for j in range(dim):
        x[i,j]=x_min+ (x_max-x_min) * random()
        v[i,j]=ini_v
        y[i,j]=1e10
    
t=0
    
while(generation < 100):
    cp=ax.contourf(x1, x2, z, 20, cmap='viridis') 
    for i in range(part):
        fx[i,0]=rastrigin(x[i,:])
                
        if fx[i,0]<f_ind[i,0]:
            y[i,:]=x[i,:]
            f_ind[i,0]=fx[i,0]    
    
    
    bestfitness=np.amin(f_ind)
    result = np.where(f_ind == np.amin(f_ind))
    p=result[0]
    
    ys=y[p,:]#posição da melhor partícula
    
    for j in range(dim):
        for i in range(part):             
            r1=random()
            r2=random()
                      
            v[i,j]=w*v[i,j] + c1*r1*(y[i,j]-x[i,j]) + c2*r2*(ys[0,j]-x[i,j])
    
            if math.fabs(v[i,j])>max_v:
                if v[i,j]>0:
                    v[i,j] = max_v
                else:
                    v[i,j] = -max_v
                    
            x[i,j]=x[i,j] + v[i,j]
          
    w=w+slope    
    
    generation=generation+1
    
    print('Generacion: ' + str(generation) + ' - - - Gbest: ' +str(ys[0,:]))
    
    line1 = ax.plot(x[:,0], x[:,1], 'ro', linewidth=2, markersize=4)
    line2 = ax.plot(ys[0,0],ys[0,1], 'o',color='orange', linewidth=2, markersize=4)    
    
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
   
    #fig.canvas.draw()
    plt.pause(0.1)
    ax.clear()
    ax.grid(True)
