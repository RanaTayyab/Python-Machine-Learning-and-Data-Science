import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt



def sigmoid(three):
    out = 1.0 + np.exp(-1.0 * three)
    
    d = 1.0 / out
    
    return d


def costEvaluationHerewithLambda(ValueofTheta, x, y, lambdavalues):
    ValueofTheta = np.matrix(ValueofTheta)
    
    x = np.matrix(x)
    
    y = np.matrix(y)

    h = sigmoid(np.dot(x, ValueofTheta.T))

    startValue = np.multiply(-y, np.log(h))
    
    lastvalue = np.multiply((1 - y), np.log(1 - h))

    m = len(x)
    
    valuenexttheta = ValueofTheta[:, 1:ValueofTheta.shape[1]]
    

    regular = (lambdavalues / 2 * m * np.sum(np.power(valuenexttheta, 2)))

    newvari = np.sum(startValue - lastvalue)/m
    

    ans = newvari + regular

    return ans

def theXvaluesAreMapped(x1, x2, powerIs):

    theXvalues = np.ones(shape=(x1.size, 1))

    d = powerIs+1
    temp2 = 0

    for i in range(1, d):
        for j in range(i+1):

            newvari = ((x1 ** (i-j)) * (x2 ** j))
            
            newvari = np.matrix(newvari)
            
            
            newvari = np.matrix.reshape(newvari, x1.size, 1)
            
            
            theXvalues = np.append(theXvalues, newvari, axis=1)
            

    return theXvalues


def UpdatedVersionOfGradientDesc(ValueofTheta, x, y, lambdavalues):
    ValueofTheta = np.matrix(ValueofTheta)
    
    x = np.matrix(x)
    
    y = np.matrix(y)

    newvalueshaving = int(ValueofTheta.ravel().shape[1])
    
    deltaval = np.zeros(newvalueshaving)
    
    
    h = sigmoid(np.dot(x, ValueofTheta.T))
    error = h - y

    for i in range(newvalueshaving):
        
        
        evaluatingTerm = np.multiply(error, x[:, i])

        if i == 0:
            
            
            deltaval[i] = np.sum(evaluatingTerm) / len(x)
            
            
        else:
            
            
            deltaval[i] = (np.sum(evaluatingTerm) / len(x)) + ((lambdavalues / len(x)) * ValueofTheta[:, i])
            
            

    return deltaval


pandasFrame = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2', 'Acceptance'])

X = pandasFrame.as_matrix(columns=['Test 1', 'Test 2'])

Y = pandasFrame.as_matrix(columns=['Acceptance'])

placing = np.where(Y == 1)[0]

notplace = np.where(Y == 0)[0]

plt.scatter(X[placing, 0], X[placing, 1], marker='+', s=40, c='black', label='Accepted')

plt.scatter(X[notplace, 0], X[notplace, 1], marker='o', s=40, c='yellow', label='Not Accepted')

plt.xlabel('M1')


plt.ylabel('M2')
plt.legend(['y = 1', 'y = 0'])


plt.axis([-1, 1.5, -0.8, 1.2])
plt.show()

x1 = X[:, 0]

powerIs = 6

x2 = X[:, 1]

lambdavalues = 1

matrixwithValuesAre = theXvaluesAreMapped(x1, x2, powerIs)



startval = np.zeros(shape=(matrixwithValuesAre.shape[1]))


StCost = costEvaluationHerewithLambda(startval, matrixwithValuesAre, Y, lambdavalues)



answerWeGot = opt.fmin_tnc(func=costEvaluationHerewithLambda, x0=startval, fprime=UpdatedVersionOfGradientDesc, args=(matrixwithValuesAre, Y, lambdavalues))

RegThetaAfterOpti = np.array(answerWeGot[0])



OpCost = costEvaluationHerewithLambda(RegThetaAfterOpti, matrixwithValuesAre, Y, lambdavalues)



one = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)

two = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)

three = np.zeros(shape=(len(one), len(two)))

for i in range(len(one)):

    for j in range(len(two)):
        
        testingNew = theXvaluesAreMapped(np.array(one[i]), np.array(two[j]), powerIs)
        
        three[i, j] = testingNew.dot(np.array(RegThetaAfterOpti))

three = three.T

plt.scatter(X[placing, 0], X[placing, 1], marker='+', s=40, c='black', label='Accepted')

plt.scatter(X[notplace, 0], X[notplace, 1], marker='o', s=40, c='yellow', label='Not Accepted')

plt.contour(one, two, three, levels=[0], colors='g')

plt.xlabel('Micro T1')

plt.ylabel('Micro T2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'])

plt.show()

print("Cost value we get after optimization process: " + str(round(OpCost, 3)))

print("Theta Values we get after optimization process: \n ")

print(RegThetaAfterOpti)
