import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

x_tr = []
y_tr = []

def func(x, a, b, c, d, e, f):
    return a * np.exp(-b * x) * np.sin(c*x+d) + e*x + f;
figure = plt.figure()
plt.plot(x_tr, y_tr, 'o', label='data')

xfuture = np.linspace(0,31,2100)
x = np.linspace(0,21,2000)
xdata = np.array(x_tr)
ydata = np.array(y_tr)


popt, pcov = curve_fit(func, xdata, ydata)
print('The a, b, c, d, e, f values respectively for function a * exp(-b * x) * sin(c*x+d) + e*x + f:\n' , popt,'\n')
plt.plot(xfuture, func(xfuture, *popt), 'r-', label='Prediction')

plt.plot(x, func(x, *popt), 'b-',
          label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f, f=%5.3f' % tuple(popt))

plt.ylim(ymax=1)


residuals = ydata- func(xdata, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res / ss_tot)
print('R squred value: \n' , r_squared,'\n')

plt.title('THIS IS THE TITLE')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
figure.savefig("figCurve.png")