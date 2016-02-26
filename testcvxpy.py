from cvxpy import *
import numpy

x = Variable(5)
y = Variable(5)

obj = 0

constrain = []

for i in range(x.size[0]):
    obj += square(x[i]+0.5*y[i]+1)
    constrain.append(x[i]+y[i] == 1)

problem = Problem(Minimize(obj), constrain)
problem.solve()
print problem.value
print x.value
print y.value
