from utils import *
from plots import * 
from expectation_values import *
from numpy import tanh
x=np.linspace(0.0001,1,10)
z1=0.5
x1=0
y1=[0.5*z1/tanh(1/(2*t)) - 1.0 + 0.5/(z1*tanh(1/(2*t))) for t in x]
y2= [sqrt(-(0.5*z1/tanh(1/(2*t)) - 1.0 + 0.5/(z1*tanh(1/(2*t))))**2 + 1.0*(0.5*z1**4 - 1.0*z1**3*tanh(1/(2*t)) + 0.5*z1**2*tanh(1/(2*t))**2 + 0.5*z1**2 - 1.0*z1*tanh(1/(2*t)) + 0.5)/(z1**2*tanh(1/(2*t))**2)) for t in x]
y3=[1.41421356237309*z1*(0.5*z1**2 - 1.0*z1*tanh(1/(2*t)) + 0.5)/(sqrt((0.5*z1**4 - z1**2*tanh(1/(2*t))**2 + 0.5)/tanh(1/(2*t))**2)*tanh(1/(2*t))*abs(z1)) for t in x]
plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.show()