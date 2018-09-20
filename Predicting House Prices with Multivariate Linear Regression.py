
# coding: utf-8

# In[139]:


#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m1,m2, xarr1,xarr2,yarr):
    totalError = 0
    for i in range(0, len(yarr)):
        x1 = xarr1[i]
        x2 = xarr2[i]
        y = yarr[i]
        totalError += (y - ((m1 * x1) + (m2 * x2) + b)) ** 2
    return totalError / float(len(yarr))

def step_gradient(b_current, m1_current, m2_current, xarr1,xarr2,yarr, learningRate):
    b_gradient = 0
    m1_gradient = 0
    m2_gradient = 0
    N = float(len(yarr))
    for i in range(0, len(yarr)):
        x1 = xarr1[i]
        x2 = xarr2[i]
        y = yarr[i]
        b_gradient += -(2/N) * 1 * (y - ((m1_current * x1) + (m2_current * x2) + b_current))
        m1_gradient += -(2/N) * x1 * (y - ((m1_current * x1) + (m2_current * x2) + b_current))
        m2_gradient += -(2/N) * x2 * (y - ((m1_current * x1) + (m2_current * x2) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m1 = m1_current - (learningRate * m1_gradient)
    new_m2 = m2_current - (learningRate * m2_gradient)
    return [new_b, new_m1, new_m2]

def gradient_descent_runner(x1,x2,y, starting_b, starting_m1, starting_m2, learning_rate, num_iterations):
    b = starting_b
    m1 = starting_m1
    m2 = starting_m2
    j=[]
    for i in range(num_iterations):
        b, m1, m2 = step_gradient(b, m1, m2, x1,x2,y, learning_rate)
        e= compute_error_for_line_given_points(b, m1, m2, x1,x2,y)
        j.append(e)
    v = array(j)
    v2=arange(num_iterations)
    plt.ylabel('Cost J')
    plt.xlabel('No of Iterations')
    plt.plot(v2,v)
    plt.show()
    return [b, m1, m2]

def run(x1,x2,y):
    learning_rate = 0.01
    initial_b = 0 # initial y-intercept guess
    initial_m1 = 0 # initial slope guess
    initial_m2 = 0
    num_iterations = 2000
    be= compute_error_for_line_given_points(initial_b, initial_m1, initial_m2, x1,x2,y)
    print("Previously: Slope1, Slope2, Y-Intercept & Error")
    print(initial_m1)
    print(initial_m2)
    print(initial_b)
    print(be)
    [b, m1,m2] = gradient_descent_runner(x1,x2,y, initial_b, initial_m1,initial_m2, learning_rate, num_iterations)
    print("\nAfter Training: Slope1, Slope2, Y-Intercept & Error")
    print(m1)
    print(m2)
    print(b)
    e= compute_error_for_line_given_points(b, m1,m2, x1,x2,y)
    #print(e)
    return [b,m1,m2]

def Normalize(a,b,c):
    av1 = np.mean(a)
    av2 = np.mean(b)
    av3 = np.mean(c)
   
    
    sa = np.subtract(a, av1)
    sb = np.subtract(b,av2)
    sc = np.subtract(c,av3)
   
    
    sta = np.std(a)
    stb = np.std(b)
    stc = np.std(c)
    
    
    new_a = np.divide(sa,sta)
    new_b = np.divide(sb,stb)
    new_c = np.divide(sc,stc)
    
    
    return [new_a, new_b, new_c]
    
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import style
    import numpy as np
    style.use('ggplot')
    a,b,c = np.loadtxt('ex1data2.txt', unpack=True, delimiter=',')
    x1, x2, y = Normalize(a,b,c)
    
    plt.scatter(x1,c, marker="x")
    plt.scatter(x2,c)
    plt.title('Scatter Plot')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')
    plt.show()
    [b,m1,m2] = run(x1,x2,c)
    c= m1*x1+m2*x2+b
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, c, c='r', marker='o')
    ax.set_xlabel('Area of House')
    ax.set_ylabel('No of Rooms')
    ax.set_zlabel('Price of House')
    plt.show()
    
    newdata = (m1*((1650-2000.680)/786.202))+(m2*((3-3.170)/0.752))+b
    print("\n\nPredicted price is: ")
    print(newdata)

