import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Example data approximating the dashed curve
x_data = np.linspace(1, 15, 100)
y_data=[0.76885193, 0.79889978, 0.82356021, 0.84340478, 0.86048416, 0.87463067, 0.88686654, 0.89728291, 0.90641398, 0.91424963, 0.92129944, 0.92731626, 0.93277011, 0.93762027, 0.9419125, 0.94583119, 0.94933797, 0.95249005, 0.95538864, 0.95803147, 0.96040557, 0.96260376, 0.96463998, 0.9664924, 0.96819493, 0.96978193, 0.97126218, 0.97262433, 0.97388252, 0.97506255, 0.97616999, 0.97721003, 0.97816562, 0.97906502, 0.97991365, 0.98071488, 0.98147187, 0.98217555, 0.98283904, 0.98346839, 0.98406569, 0.98463292, 0.9851719, 0.98567463, 0.98615344, 0.98660992, 0.98704533, 0.98746086, 0.98785762, 0.98823358, 0.98858975, 0.98893078, 0.98925747, 0.98957055, 0.98987072, 0.99015865, 0.99043437, 0.99069548, 0.99094653, 0.99118798, 0.9914203, 0.99164392, 0.99185923, 0.99206664, 0.99226649, 0.9924567, 0.99263996, 0.99281689, 0.99298775, 0.9931528, 0.9933123, 0.99346648, 0.99361556, 0.99375976, 0.99389818, 0.99403152, 0.99416069, 0.99428585, 0.99440717, 0.99452479, 0.99463885, 0.9947495, 0.99485685, 0.99496104, 0.99506199, 0.99515906, 0.99525339, 0.99534507, 0.9954342, 0.99552088, 0.99560518, 0.99568719, 0.99576698, 0.99584465, 0.99592025, 0.99599386, 0.99606524, 0.99613433, 0.99620167, 0.99626732]

# Candidate function: Logistic function
def logistic(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))

# Fit the logistic function to the data
params, params_covariance = curve_fit(logistic, x_data, y_data, p0=[1, 1, 1])

# Generate the fitted curve
x_fit = np.linspace(1, 15, 100)
y_fit = logistic(x_fit, *params)

# Plot the original data, tanh function, and the fitted curve
plt.plot(x_data, y_data, 'b--', label='Dashed Curve Data')
plt.plot(x_fit, np.tanh(x_fit), 'r-', label='tanh(x) Function')
plt.plot(x_fit, y_fit, 'g-', label='Logistic Fit')
plt.legend()
plt.show()

# Print fitted parameters
print("Fitted parameters:", params)