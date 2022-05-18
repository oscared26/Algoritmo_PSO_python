import numpy as np
import random as rand
import matplotlib.pyplot as plt
from random import random, seed
import math

import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter

import matplotlib.animation as animation

#função a ser estudada
def sphere(x):
    d = x.shape[0]
    sum = 0
    for i in range(d):
        sum = sum + x[i] ** 2

    return sum

# limites para a função Sphere
x_max = 10
x_min = -10


# Graficar a superficie da função sphere
xx = np.linspace(x_min, x_max, 200)
yy = np.linspace(x_min, x_max, 200)
# create grid
x1, x2 = np.meshgrid(xx, yy)

l = [x1, x2]
arr = np.array(l)

z = sphere(arr)


#Parâmetros do algoritmo PSO
it = 1
c1 = 2.05
c2 = 2.05
ini_v = 3

itmax = 100
wmax = 0.9
wmin = 0.4
max_v = ini_v / 3

w = wmax - it * (wmax - wmin) / itmax


# número de partículas
S = 20
# número de dimensões
d = 2
# posição de cada partícula
x = np.zeros((S, d))
# velocidade de cada partícula
v = np.zeros((S, d))
# posição de melhor localização de cada partícula
Pbest = np.zeros((S, d))
# o custo da i-ésima partícula
fitness = np.zeros((S, 1))
# O melhor fitness local visitado pela i-ésima partícula
Pbest_fitness = 1e10 * np.ones([S, 1])

#Array par armazenar a posição das partículas em cada iteração
zarray = np.zeros((S, d, itmax))

# Inicialização de x, v e Pbest
for i in range(S):
    for j in range(d):
        x[i, j] = x_min + (x_max - x_min) * random()
        v[i, j] = ini_v

Pbest = np.copy(x)

while (it < itmax):

    # Para cada partícula
    for i in range(S):
        # Avalie o fitness da função objetivo
        fitness[i, 0] = rastrigin(x[i, :])
        # Encontra o melhor fitness e a posição
        if fitness[i, 0] < Pbest_fitness[i, 0]:
            Pbest[i, :] = x[i, :]
            Pbest_fitness[i, 0] = fitness[i, 0]

            # Encontra o melhor fitness da população
    bestfitness = np.amin(Pbest_fitness)
    result = np.where(Pbest_fitness == np.amin(Pbest_fitness))
    p = result[0]

    # Posição da melhor partícula
    Gbest = Pbest[p, :]

    # Para cada partícula
    for j in range(d):
        for i in range(S):
            # Gera valores aleatórios para dar aleatoridade à busca
            r1 = random()
            r2 = random()

            # Atualiza a nova velocidade
            v[i, j] = w * v[i, j] + c1 * r1 * (Pbest[i, j] - x[i, j]) + c2 * r2 * (Gbest[0, j] - x[i, j])

            # Limita a velocidade de cada partícula
            # Estabelecendo valor máximo e mínimo da velocidade
            if math.fabs(v[i, j]) > max_v:
                if v[i, j] > 0:
                    v[i, j] = max_v
                else:
                    v[i, j] = -max_v

            # Atualiza a nova posição
            x[i, j] = x[i, j] + v[i, j]
            zarray[i, j, it] = x[i, j]

            # Limita a posição de cada partícula
            # Estabelecendo valor máximo e mínimo da posição
            if x[i, j] > x_max:
                x[i, j] = x_max
                zarray[i, j, it] = x_max

            if x[i, j] < x_min:
                x[i, j] = x_min
                zarray[i, j, it] = x_min

    # Calcula o fator de inercia
    w = wmax - it * (wmax - wmin) / itmax
    # Incremento da geração
    it = it + 1

    print('Generacion: ' + str(it) + ' - - - Gbest: ' + str(Gbest[0, :]) + ' F(x)= ' + str(bestfitness))


#Animação 
mycmap = plt.get_cmap('gist_earth')
fps = 10  # frame per sec
frn = itmax  # frame number of the animation

def create_video():
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cp = ax.contourf(x1, x2, z, 20, cmap=mycmap)
    
    def animate(frame_number, zarray, plot):
        ax.clear()
        cp = ax.contourf(x1, x2, z, 20, cmap=mycmap)
        
        #Representa as partículas do sistema
        line1 = ax.plot(zarray[:, 0, frame_number], zarray[:, 1, frame_number], 'ro', linewidth=2, markersize=4)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.grid(True)
        return line1

    ani = FuncAnimation(
        fig,
        animate,
        frn,
        fargs=(zarray, cp),
        interval=100,
        blit=True
    )

    plt.show()

    return ani

anim = create_video()
fn = 'plot_surface_animation_funcanimation'
anim.save(fn+'.mp4',writer='ffmpeg',fps=fps)
anim.save(fn+'.gif',writer='imagemagick',fps=fps)
