import numpy as np
from numpy import linspace, sqrt

import matplotlib.pyplot as plt

z=1

nu=linspace(1,5,100)

#fock state
N= 0.25*(nu**2*z**4 + nu**2*z**2 + nu**2 + nu*z**3 + nu*z - z**2)/(z*(0.25*nu + z*(0.25*nu*z + 0.5))) 
deltasq= (4.0*sqrt((-0.015625*(0.25*nu + z*(0.25*nu*z + 0.5))*(nu**2*z**4 + nu**2*z**2 + nu**2 + nu*z**3 + nu*z - z**2)**2 + (0.5*nu + z*(0.5*nu*z + 1))**2*(0.0234375*nu**3*z**6 + 0.0234375*nu**3*z**4 + 0.0234375*nu**3*z**2 + 0.0234375*nu**3 + 0.015625*nu**2*z**5 + 0.015625*nu**2*z**3 + 0.015625*nu**2*z - 0.03125*nu*z**4 - 0.03125*nu*z**2 - 0.015625*z**3))/((0.25*nu + z*(0.25*nu*z + 0.5))*(0.5*nu + z*(0.5*nu*z + 1))**2))*abs(z)/z**2 )**2

plt.plot(nu,N)
plt.plot(nu,deltasq)
plt.legend(['N','delta^2'])
plt.show()




z=linspace(0,1,100)
nu=3
snr_ext_gauss= 1.41421356237309*nu*(z*(0.5*z - 1.0) + 0.5)*abs(z)/(z*sqrt(0.5*nu**2*z**4 + 0.5*nu**2 - z**2))
snr_ext_photonadd= 0.25*z*(0.25*nu**2*z**4 - 0.25*nu**2*z**3 + 0.25*nu**2*z**2 - 0.25*nu**2*z + 0.25*nu**2 + 0.5*nu*z**3 - 0.5*nu*z**2 + 0.5*nu*z + 0.25*z**2)/(sqrt((-0.015625*(0.25*nu + z*(0.25*nu*z + 0.5))*(nu**2*z**4 + nu**2*z**2 + nu**2 + nu*z**3 + nu*z - z**2)**2 + (0.5*nu + z*(0.5*nu*z + 1))**2*(0.0234375*nu**3*z**6 + 0.0234375*nu**3*z**4 + 0.0234375*nu**3*z**2 + 0.0234375*nu**3 + 0.015625*nu**2*z**5 + 0.015625*nu**2*z**3 + 0.015625*nu**2*z - 0.03125*nu*z**4 - 0.03125*nu*z**2 - 0.015625*z**3))/((0.25*nu + z*(0.25*nu*z + 0.5))*(0.5*nu + z*(0.5*nu*z + 1))**2))*(0.25*nu + z*(0.25*nu*z + 0.5))*abs(z))
snr_ext_photonsub= 0.666666666666667*z*(0.25*nu**2*z**4 - 0.25*nu**2*z**3 + 0.25*nu**2*z**2 - 0.25*nu**2*z + 0.25*nu**2 - 0.5*nu*z**3 + 0.5*nu*z**2 - 0.5*nu*z + 0.25*z**2)/(sqrt((-(0.25*nu + z*(0.25*nu*z - 0.5))*(0.333333333333333*nu**2*z**4 + 0.333333333333333*nu**2*z**2 + 0.333333333333333*nu**2 - 1.0*nu*z**3 - 1.0*nu*z + 1.0*z**2)**2 + (0.5*nu + z*(0.5*nu*z - 1))**2*(0.166666666666667*nu**3*z**6 + 0.166666666666667*nu**3*z**4 + 0.166666666666667*nu**3*z**2 + 0.166666666666667*nu**3 - 0.555555555555556*nu**2*z**5 - 0.555555555555556*nu**2*z**3 - 0.555555555555556*nu**2*z + 0.666666666666667*nu*z**4 + 0.666666666666667*nu*z**2 - 0.333333333333333*z**3))/((0.25*nu + z*(0.25*nu*z - 0.5))*(0.5*nu + z*(0.5*nu*z - 1))**2))*(0.25*nu + z*(0.25*nu*z - 0.5))*abs(z))
plt.plot(z,N)
plt.plot(z,deltasq)
plt.legend(['N','deltsq'])
plt.show()

plt.plot(z,snr_ext_gauss)
plt.plot(z,snr_ext_photonadd)
plt.plot(z,snr_ext_photonsub)

plt.legend(['gauss','photonadd','photonsub'])
plt.show()