import numpy as np

# You may find below useful for Support Vector Machine
# More details in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#from scipy.optimize import minimize

def func(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned perceptron parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P+1, 1))
    Xt = X.T
    # Your code here
    # Answer begin
    MissPoint = True 
    timelimit = 1000
    while (MissPoint==True and timelimit>=0):
        timelimit -= 1
        MissPoint = False
        for i in range(N):
            if (float(np.dot(Xt[i,:],w[0:P,0])+w[P,0])*y[0,i] <= 0):
                MissPoint = True                
                for j in range(P):
                    w[j,0] = w[j,0] + Xt[i,j]*y[0,i]
                w[P,0] = w[P,0]+y[0,i]
                
    if timelimit<0:
        print("out of limit")
    #print(w)   
    # Answer end
   
    return w
'''
X = np.mat([[1,2,5,4,6,1],[4,9,6,5,0.7,1.5]])
y = np.mat([-1,1,1,1,-1,-1])
func(X,y)
'''

