import numpy as np
from math import comb as comb
#graphviz pour les graphes
#Using Graphviz and Anytree : https://github.com/xflr6/graphviz
#voir vers la fin de https://medium.com/swlh/making-data-trees-in-python-3a3ceb050cfd

#fonction testée, elle fonctionne parfaitement,
#getSt_N(n,hn,bn,s0) renvoie un np.array de toutes les valeurs que peut prendre la variable aléatoire St_N
def getSt_N(n,hn,bn,s0):
    res = np.ones(n+1)*s0
    for i in range(n+1):
        res[i] = res[i] * np.power((1+hn),n-i) * np.power((1+bn),i)
    return res



###################################################################
####                       Question 3                          ####
###################################################################


def g(x, N, f, s, hn, bn):
    return f(s*(pow((1+hn),x))*(pow((1+bn),(N-x))))


def pricer_1(N, s, rn, hn, bn, f):
    Eq=0
    qn=(rn-bn)/(hn-bn)
    for i in range(N+1):
        Eq+=g(i, N, f, s, hn, bn)*comb(N,i)*pow(qn,i)*pow((1-qn),N-i)
    pricer1=Eq/pow((1+rn),N)
    return pricer1


###################################################################
####                       Question 4                          ####
###################################################################


def f1(x):
    return max(x-110,0)

def f2(x):
    return max(x-100,0)

pricer_1(3,100,0.02,0.05,-0.05,f1)
pricer_1(20,100,0.02,0.05,-0.05,f1)

pricer_1(20,100,0.02,0.05,-0.05,f2)
pricer_1(3,100,0.02,0.05,-0.05,f2)





###################################################################
####                       Question 5                          ####
###################################################################
def pricer_2(N,rn,hn,bn,s,f):
    qn = (rn - bn)/(hn - bn)
    print("qn = ",qn)

    St_N = getSt_N(N,hn,bn,s)
    print("St_N = ",St_N)
    Vk = f(St_N) #VN
    print("Vk = ",Vk)
    for n in range(N,0,-1):
        Vkmoins1 = np.zeros(n)
        for k in range(0,n): #on parcourt Vk
            Vkmoins1[k] = (qn * Vk[k] + (1-qn) * Vk[k+1]) / (1 +rn) # Vkmoins1 = (qn * Vk((1+hn)*St_kmoins1) + (1-qn) * Vk((1+bn)*St_kmoins1) ) / (1 + rn)

        print("St_kmoins1 = ",getSt_N(n-1,hn,bn,s))
        print("Vkmoins1 = ",Vkmoins1)
        Vk = Vkmoins1
        print("Vk = ",Vk)
    return Vkmoins1


###################################################################
####                       Question 6                          ####
###################################################################

def f(x) :
    return np.maximum(x-100,0)

pricer_2(3,0.02,0.05,-0.05,100,f)

#from anytree import Node, RenderTree
#from anytree.exporter import DotExporter
#s0 = Node("100") #root
#s1_1 = Node("95", parent=s0)
#s1_2 = Node("105", parent=s0)
#s2_1 = Node("90.25", parent=s1_1)
#s2_2 = Node("99.75", parent=s1_2)
#s2_2 = Node("99.75", parent=s1_1)
#s2_3 = Node("110.25", parent=s1_2)
#DotExporter(s0).to_picture("Sti.png")



###################################################################
####                       Question 7                          ####
###################################################################


def compare(p1, p2, N, rn, hn, bn, s, f) :
    return p1(N, rn, hn, bn, s, f) - p2(N, rn, hn, bn, s, f)

compare(pricer_1, pricer_2, np.random.randint(5,16), 0.01, 0.05, -0.05, 100, f1)


###################################################################
####                       Question 10                         ####
###################################################################


num0_a = f2((1 + 0.05) * 100) - f2((1 - 0.05) * 100)
den0_a = 100 * (0.05 + 0.05)
alpha0 = num0_a / den0_a

num0_b = f2((1 - 0.05) * 100) * (1 + 0.05) - f2((1 + 0.05) * 100) * (1 - 0.05)
den0_b = (0.05 + 0.05) * (1 + 0.03)
beta0 = num0_b / den0_b
