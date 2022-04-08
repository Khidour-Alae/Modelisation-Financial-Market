import numpy as np
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

# Question 5
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

def f(x):
    return np.maximum(x-110,0)

pricer_2(3,0.02,0.05,-0.05,100,f)