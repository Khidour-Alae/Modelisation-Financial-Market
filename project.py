import numpy as np
from math import comb as comb #need python 3.8
from scipy.stats import norm as norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # Fonction pour la 3D

#graphviz pour les graphes
#Using Graphviz and Anytree : https://github.com/xflr6/graphviz
#voir vers la fin de https://medium.com/swlh/making-data-trees-in-python-3a3ceb050cfd

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
    return np.maximum(x-110,0)

pricer_1(20,100,0.02,0.05,-0.05,f1)

###################################################################
####                       Question 5                          ####
###################################################################
def pricer_2(N,rn,hn,bn,s,f):
    qn = (rn - bn)/(hn - bn)

    St_N = getSt_N(N,hn,bn,s)
    Vk = f(St_N) #VN
    for n in range(N,0,-1):
        Vkmoins1 = np.zeros(n)
        for k in range(0,n): #on parcourt Vk
            Vkmoins1[k] = (qn * Vk[k] + (1-qn) * Vk[k+1]) / (1 +rn) # Vkmoins1 = (qn * Vk((1+hn)*St_kmoins1) + (1-qn) * Vk((1+bn)*St_kmoins1) ) / (1 + rn)
        Vk = Vkmoins1
    return Vk[0]


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
    return p1(N, s, rn, hn, bn, f) - p2(N, rn, hn, bn, s, f)

compare(pricer_1, pricer_2, np.random.randint(5,16), 0.01, 0.05, -0.05, 100, f)


###################################################################
####                       Question 10                         ####
###################################################################


def pricer_2_bis(N, rn, hn, bn, s, f, e): #cette fonction a un paramètre en plus par rapport à pricer_2, le paramètre en plus est e qui correspond à l'étape à laquelle nous voulons nous arréter
    qn = (rn - bn)/(hn - bn)

    St_N = getSt_N(N,hn,bn,s)
    Vk = f(St_N) #VN
    for n in range(N,e,-1):
        global Vkmoins1
        Vkmoins1 = np.zeros(n)
        for k in range(0,n): #on parcourt Vk
            Vkmoins1[k] = (qn * Vk[k] + (1-qn) * Vk[k+1]) / (1 +rn) # Vkmoins1 = (qn * Vk((1+hn)*St_kmoins1) + (1-qn) * Vk((1+bn)*St_kmoins1) ) / (1 + rn)
        Vk = Vkmoins1
    return Vkmoins1


num0_a = pricer_2_bis(2, 0.03, 0.05, -0.05, 100, f, 1)[0] - pricer_2_bis(2, 0.03, 0.05, -0.05, 100, f, 1)[1]
den0_a = 100 * (0.05 + 0.05)
alpha0 = num0_a / den0_a

print("alpha0 = ", alpha0)

num0_b = pricer_2_bis(2, 0.03, 0.05, -0.05, 100, f, 1)[1] * (1 + 0.05) - pricer_2_bis(2, 0.03, 0.05, -0.05, 100, f, 1)[0] * (1 - 0.05)
den0_b = (0.05 + 0.05) * (1 + 0.03)
beta0 = num0_b / den0_b

print("beta0 = ", beta0)

num1_a_1 = f((1 + 0.05) * getSt_N(1, 0.05, -0.05, 100)[0]) - f((1 - 0.05) * getSt_N(1, 0.05, -0.05, 100)[0])
den1_a_1 = 100 * (0.05 + 0.05)
alpha1_1 = num1_a_1 / den1_a_1

num1_a_2 = f((1 + 0.05) * getSt_N(1, 0.05, -0.05, 100)[1]) - f((1 - 0.05) * getSt_N(1, 0.05, -0.05, 100)[1])
den1_a_2 = 100 * (0.05 + 0.05)
alpha1_2 = num1_a_2 / den1_a_2

alpha1 =  np.array([alpha1_1, alpha1_2])
print("alpha1 = ", alpha1)

num1_b_1 = f((1 - 0.05) * getSt_N(1, 0.05, -0.05, 100)[0]) * (1 + 0.05) - f((1 + 0.05) * getSt_N(1, 0.05, -0.05, 100)[0]) * (1 - 0.05)
den1_b_1 = (0.05 + 0.05) * (1 + 0.03) * (1 + 0.03)
beta1_1 = num1_b_1 / den1_b_1


num1_b_2 = f((1 - 0.05) * getSt_N(1, 0.05, -0.05, 100)[1]) * (1 + 0.05) - f((1 + 0.05) * getSt_N(1, 0.05, -0.05, 100)[1]) * (1 - 0.05)
den1_b_2 = (0.05 + 0.05) * (1 + 0.03) * (1 + 0.03)
beta1_2 = num1_b_2 / den1_b_2

beta1 = np.array([beta1_1, beta1_2])
print("beta1 = ", beta1)

###################################################################
####                       Question 12                         ####
###################################################################

def pricer_MC(n, s, r, sigma, T, f):
    Epsi = np.normal(loc=0, scale=1, size=n)
    s = 0
    for i in range(1,n+1):
        s = s + np.exp(-r*T) * f(s * np.exp((r - sigma*sigma/2)*T + sigma*np.sqrt(T)*Epsi[i]))
    return s/n

###################################################################
####                       Question 13                         ####
###################################################################



###################################################################
####                       Question 15                         ####
###################################################################


def F(x):
    return norm.cdf(x)

def put_BS(s,r,sigma,T,K):
    d1 = (1/(sigma*np.sqrt(T)))*(np.log(s/K)+T*(r+(sigma*sigma)/2))
    d2 =  d1 - sigma*np.sqrt(T)
    prix_BS = -s*F(-d1)+K*numpy.exp(-r*T)*F(-d2)
    return prix_BS

###################################################################
####                       Question 16                        ####
###################################################################

prixFF = put_BS(100,0.01,0.1,1,90)
print("Le prix du pricer par formule fermé pour r = 0.01, sigma = 0.1, s = 100, T = 1, K = 90 est : ", prixFF)


###################################################################
####                       Question 18                         ####
###################################################################

tab_k = np.arange(1,11)
tab_T = np.array([1,1/2,1/3,1/4,1/6,1/12])
len_s = tab_s.size
len_T = tab_T.size

x = np.zeros(len_s*len_T)
y = np.zeros(len_s*len_T)
z = np.zeros(len_s*len_T)
for i in range(len_T):
    x[6*i:6*(i+1)] = tab_k[i]*np.ones(6)
    y[6*i:6*(i+1)] = np.copy(tab_T)
for i in range(len_s*len_T):
    z[i] = put_FF(20*x[i],0.01,0.1,y[i],100)

# Tracé du résultat en 3D
fig = plt.figure()
ax = fig.gca(projection='3d')  # Affichage en 3D
ax.scatter(x, y, z, label='Courbe', marker='d')  # Tracé des points 3D
plt.title("Points 3D")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()    
    


