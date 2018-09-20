
# coding: utf-8

# In[14]:


#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("ex1data1.txt", delimiter=",")
    learning_rate = 0.01
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    be= compute_error_for_line_given_points(initial_b, initial_m, points)
    print("Previously: Slope, Y-Intercept & Error")
    print(initial_m)
    print(initial_b)
    print(be)
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("\nAfter Training: Slope, Y-Intercept & Error")
    print(m)
    print(b)
    e= compute_error_for_line_given_points(b, m, points)
    print(e)
    return [b,m]
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import style
    import numpy as np
    style.use('ggplot')
    x,y= np.loadtxt('ex1data1.txt', unpack=True, delimiter=',')
    plt.scatter(x,y,marker="x")
    plt.title('Scatter Plot')
    plt.ylabel('Y axis')
    plt.xlabel('X axis')
    [b,m] = run()
    y= m*x+b
    plt.plot(x,y,color='blue')
    plt.show()

