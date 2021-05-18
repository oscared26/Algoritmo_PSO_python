import numpy as np
from random import random,seed
import math

#seed(1)

def sphere(x):
  d=x.shape[0]
  sum=0
  for i in range(d):
    sum=sum+x[i]**2
  
  return sum

x_max=8
x_min=-8




c1=2
c2=2
ini_v=0.5

maxiter = 50
w0 = 0.9
wf=0.1
slope = (wf-w0)/maxiter
w= w0
max_v = 5

S=5
x=np.zeros((S,2))
v=np.zeros((S,2))
y=np.zeros((S,2))

part,dim=x.shape


f_ind=1e10*np.ones([part,1])
fx=np.zeros((part,1))

for i in range(part):
  for j in range(dim):
    x[i,j]=x_min+ (x_max-x_min) * random()
    v[i,j]=ini_v
    y[i,j]=1e10

t=0
while t<maxiter:

  for i in range(part):
    fx[i,0]=sphere(x[i,:])
    if fx[i,0]<f_ind[i,0]:
      y[i,:]=x[i,:]
      f_ind[i,0]=fx[i,0]


  bestfitness=np.amin(f_ind)
  result = np.where(f_ind == np.amin(f_ind))
  p=result[0]

  ys=y[p,:]

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
  t=t+1

print("Gets the global best fitness=> {:.5f}".format(sphere(ys[0,:])))
print("display the solution of the optimization problem x1=> {:.5f} x2=> {:.5f}".format(ys[0,0],ys[0,1]))
