# 3D Cubic Bezier Curve

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import matplotlib.animation as animation

# User Adefined variables, x represents the 3 x values of the 3 points
# x = np.random.random_sample((4, ))
# y = np.random.random_sample((4, ))
# z = np.random.random_sample((3, ))
x = np.array([0.1, 0.3, 0.8, 1])
y = np.array([0, 0.9, 0.6, 0.1])
z = np.array([0.1, 0.9, 0.2, 0.7])

CELLS = 100  # Total number of division for bezier curve

# Other variables
nCPTS = np.size(x, 0)  # Total number of control points, should be 3

n = nCPTS - 1  # Total number of segments, 2 for 2d
i = 0  # Central point counter
t = np.linspace(0, 1, CELLS)  # Parametrix variable
b = []  # Initialized empty matrix for Bernstein Basis polynomial

# Initialized empty matrix for X, Y, and Z values of bezier curve
xBezier = np.zeros((1, CELLS))
yBezier = np.zeros((1, CELLS))
zBezier = np.zeros((1, CELLS))
# zBezier = np.zeros((1, CELLS))
secondaryPointsX = np.zeros((3, CELLS))
secondaryPointsY = np.zeros((3, CELLS))
secondaryPointsZ = np.zeros((3, CELLS))
tertiaryPointsX = np.zeros((2, CELLS))
tertiaryPointsY = np.zeros((2, CELLS))
tertiaryPointsZ = np.zeros((2, CELLS))


# Binomial coefficients = n choose i
def Ni(n, i):
    return np.math.factorial(n) / (np.math.factorial(i) *
                                   np.math.factorial(n - i))


# Bernstein Basis polynomial
def basisFunction(n, i, t):
    J = np.array(Ni(n, i) * (t ** i) * (1 - t) ** (n - i))
    return J


# Main loop, k represents the index of each control point
for k in range(0, nCPTS):
    b.append(basisFunction(n, i, t))

    # Bezier curve calculation
    xBezier = basisFunction(n, i, t) * x[k] + xBezier
    yBezier = basisFunction(n, i, t) * y[k] + yBezier
    zBezier = basisFunction(n, i, t) * z[k] + zBezier
    i += 1

for i in range(0, nCPTS - 1):
    secondaryPointsX[i] = (1 - t) * x[i] + t * x[i + 1]
    secondaryPointsY[i] = (1 - t) * y[i] + t * y[i + 1]
    secondaryPointsZ[i] = (1 - t) * z[i] + t * z[i + 1]
for i in range(0, nCPTS - 2):
    tertiaryPointsX[i] = (
                                 (1 - t) ** 2) * x[i] + 2 * (1 - t) * t * x[i + 1] + (t ** 2) * x[i + 2]
    tertiaryPointsY[i] = (
                                 (1 - t) ** 2) * y[i] + 2 * (1 - t) * t * y[i + 1] + (t ** 2) * y[i + 2]
    tertiaryPointsZ[i] = (
                                 (1 - t) ** 2) * z[i] + 2 * (1 - t) * t * z[i + 1] + (t ** 2) * z[i + 2]

# Plotting
fig1 = plt.figure("3D Cubic Bezier Curve", figsize=(4, 4))
ax1 = plt.subplot(111, projection="3d")
ax1.set_xlabel("3D Cubic Bezier Curve", fontsize=16)
ax1.plot(x, y, z, linewidth=0.7, c="blue")
secondaryPoints, = ax1.plot([], [], [], c="#06bfb6", marker="o", linewidth=0.7, markersize=3)
tertiaryPoints, = ax1.plot([], [], [], c="green", marker="o", linewidth=0.7, markersize=3)
ax1.plot(xBezier[0], yBezier[0], zBezier[0], c="orange")
movingPoint, = ax1.plot([], [], [], marker="o", markersize=7, color="red")
ax1.scatter(x, y, z, c="black")
clockText = "t=0"
clock = ax1.text2D(1, 0, clockText, transform=ax1.transAxes)

equationTextX = "x(t) = (1-t)²(" + str(x[0]) + ") + 3t(1-t)²(" + str(x[1]) + ") + 3t²(1-t)(" + str(
    x[2]) + ") + t³(" + str(x[3]) + ")"
equationTextY = "y(t) = (1-t)²(" + str(y[0]) + ") + 3t(1-t)²(" + str(y[1]) + ") + 3t²(1-t)(" + str(
    y[2]) + ") + t³(" + str(y[3]) + ")"
equationTextZ = "z(t) = (1-t)²(" + str(z[0]) + ") + 3t(1-t)²(" + str(z[1]) + ") + 3t²(1-t)(" + str(
    z[2]) + ") + t³(" + str(z[3]) + ")"
ax1.text2D(0, 0.9, equationTextX, transform=ax1.transAxes)
ax1.text2D(0, 0.95, equationTextY, transform=ax1.transAxes)
ax1.text2D(0, 1, equationTextZ, transform=ax1.transAxes)


def animate(frame):
    movingPointX = xBezier[0][frame]
    movingPointY = yBezier[0][frame]
    movingPointZ = zBezier[0][frame]

    secondaryPoints.set_xdata([
        secondaryPointsX[0][frame], secondaryPointsX[1][frame],
        secondaryPointsX[2][frame]
    ])
    secondaryPoints.set_ydata([
        secondaryPointsY[0][frame], secondaryPointsY[1][frame],
        secondaryPointsY[2][frame]
    ])
    secondaryPoints.set_3d_properties([
        secondaryPointsZ[0][frame], secondaryPointsZ[1][frame],
        secondaryPointsZ[2][frame]
    ])

    tertiaryPoints.set_xdata(
        [tertiaryPointsX[0][frame], tertiaryPointsX[1][frame]])
    tertiaryPoints.set_ydata(
        [tertiaryPointsY[0][frame], tertiaryPointsY[1][frame]])
    tertiaryPoints.set_3d_properties(
        [tertiaryPointsZ[0][frame], tertiaryPointsZ[1][frame]])
    movingPoint.set_xdata([movingPointX])
    movingPoint.set_ydata([movingPointY])
    movingPoint.set_3d_properties([movingPointZ])

    clockText = "t=" + str((frame / 100))
    clock.set_text(clockText)
    return secondaryPoints, tertiaryPoints, movingPoint


animator = animation.FuncAnimation(fig=fig1,
                                   func=animate,
                                   frames=100,
                                   interval=100)

plt.show()
