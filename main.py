import numpy as np
import matplotlib.pyplot as plt

# User defined variables
x = np.random.random_sample((3,))
y = np.random.random_sample((3,))
z = np.random.random_sample((3,))

CELLS = 100 # Total number of division for bezier curve

# Other variables
nCPTS = np.size(x, 0) # Total number of control points
n = nCPTS - 1 # Total number of segments
i = 0 # Central point counter
t = np.linspace(0, 1, CELLS) # Parametrix variable
b = [] # Initialized empty matrix for Bernstein Basis polynomial

# Initialized empty matrix for X, Y, and Z bezier curve
xBezier = np.zeros((1, CELLS))
yBezier = np.zeros((1, CELLS))
zBezier = np.zeros((1, CELLS))

# Binomial coefficients
def Ni(n, i):
  return np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n - i))

# Bernstein Basis polynomial
def basisFunction(n, i, t):
  J = np.array(Ni(n, i) * (t ** i) * (1 - t) ** (n - i))
  return J

# Main loop
for k in range(0, nCPTS):
  b.append(basisFunction(n, i, t))
  
  # Bezier curve calculation
  xBezier = basisFunction(n, i, t) * x[k] + xBezier
  yBezier = basisFunction(n, i, t) * y[k] + yBezier
  zBezier = basisFunction(n, i, t) * z[k] + zBezier
  i += 1

# Plotting
fig1 = plt.figure(figsize = (4, 4))
ax1 = fig1.add_subplot(111, projection = "3d")
ax1.scatter(x, y, z, c = "black")
ax1.plot(xBezier[0], yBezier[0], zBezier[0], c = "blue")
plt.show()

