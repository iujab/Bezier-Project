# 2D Quadratic Bezier Curve

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import matplotlib.animation as animation


class DraggablePlotExample(object):
    def __init__(self):
        self._figure, self._axes, self._line, self._bLine = None, None, None, None
        self._dragging_point = None
        self._xpoints = [10, 30, 80]
        self._ypoints = [10, 90, 80]
        self.CELLS = 100
        self._animator = None
        self._xBezier = None
        self._yBezier = None
        self._animt = True
        self._init_plot()

    def _init_plot(self):
        self._figure = plt.figure("2D Quadratic Bezier Curve")
        self._ax1 = plt.subplot(111)
        self._ax1.set_xlabel("2D Quadratic Bezier Curve", fontsize=16)
        axes = plt.subplot(1, 1, 1)
        axes.set_xlim(0, 100)
        axes.set_ylim(0, 100)
        axes.grid(which="both")
        self._axes = axes

        self._figure.canvas.mpl_connect('button_press_event', self._on_click)
        self._figure.canvas.mpl_connect('button_release_event', self._on_release)
        self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        plt.show()
        self._update_plot()

    # Binomial coefficients
    def _Ni(self, n, i):
        return np.math.factorial(n) / (np.math.factorial(i) *
                                       np.math.factorial(n - i))

    # Bernstein Basis polynomial
    def _basisFunction(self, n, i, t):
        J = np.array(self._Ni(n, i) * (t ** i) * (1 - t) ** (n - i))
        return J

    def _update_plot(self):
        # Variable reset
        nCPTS = np.size(self._xpoints, 0)  # Total number of control points, should be 3
        n = nCPTS - 1  # Total number of segments, 2 for 2d
        i = 0  # Central point counter
        t = np.linspace(0, 1, self.CELLS)  # Parametrix variable
        b = []  # Initialized empty matrix for Bernstein Basis polynomial
        self._xBezier = np.zeros((1, self.CELLS))
        self._yBezier = np.zeros((1, self.CELLS))

        for k in range(0, nCPTS):
            b.append(self._basisFunction(n, i, t))

            # Bezier curve calculation
            self._xBezier = self._basisFunction(n, i, t) * self._xpoints[k] + self._xBezier
            self._yBezier = self._basisFunction(n, i, t) * self._ypoints[k] + self._yBezier
            i += 1

        clock = self._ax1.text(-12, 105, "")
        eqtX = self._ax1.text(0, 105, "")
        eqtY = self._ax1.text(0, 110, "")

        if not self._line:
            self._bLine, = self._ax1.plot(self._xBezier[0],
                                          self._yBezier[0],
                                          c="orange")
            self._line, = self._axes.plot(self._xpoints,
                                          self._ypoints,
                                          linewidth=0.7,
                                          c="blue",
                                          marker="o",
                                          markersize=6)
        # Update current plot
        else:
            self._line.set_data(self._xpoints, self._ypoints)
            self._bLine.set_data(self._xBezier[0], self._yBezier[0])

        secondaryPoints, = self._ax1.plot([], [], c="#06bfb6",
                                          marker="o",
                                          linewidth=0.7,
                                          markersize=3)
        movingPoint, = self._ax1.plot([], [],
                                      marker="o",
                                      markersize=7,
                                      color="red")

        def animate(frame):
            movingPointX = self._xBezier[0, frame]
            movingPointY = self._yBezier[0, frame]
            secondaryPointsX = [0, 0]
            secondaryPointsY = [0, 0]

            for i in range(0, nCPTS - 1):
                secondaryPointsX[i] = (1 - (frame / 100)) * self._xpoints[i] + (
                        frame / 100) * self._xpoints[i + 1]
                secondaryPointsY[i] = (1 - (frame / 100)) * self._ypoints[i] + (
                        frame / 100) * self._ypoints[i + 1]

            secondaryPoints.set_xdata(secondaryPointsX)
            secondaryPoints.set_ydata(secondaryPointsY)
            movingPoint.set_xdata([movingPointX])
            movingPoint.set_ydata([movingPointY])

            clockText = "t=" + str((frame / 100))
            clock.set_text(clockText)
            equationTextX = "x(t) = (1-t)³(" + str(round(self._xpoints[0], 2)) + ") + 3t(1-t)²(" + str(
                round(self._xpoints[1], 2)) + ") + 3t²(1-t)(" + str(round(self._xpoints[2], 2)) + ")"
            equationTextY = "y(t) = (1-t)³(" + str(round(self._ypoints[0], 2)) + ") + 3t(1-t)²(" + str(
                round(self._ypoints[1], 2)) + ") + 3t²(1-t)(" + str(round(self._ypoints[2], 2)) + ")"
            eqtX.set_text(equationTextX)
            eqtY.set_text(equationTextY)

            return secondaryPoints, movingPoint, clock

        if self._animt:
            self._animator = animation.FuncAnimation(fig=self._figure,
                                                     func=animate,
                                                     frames=100,
                                                     interval=100)
            self._animt = False

        self._figure.canvas.draw()

    def _add_point(self, event):
        if isinstance(event, MouseEvent):
            g, k = self._find_neighbor_point(event)
            i = self._xpoints.index(g)
            self._xpoints[i] = event.xdata
            self._ypoints[i] = event.ydata
        return event.xdata, event.ydata

    def _remove_point(self, x, _):
        if x in self._xpoints:
            i = self._xpoints.index(x)
            self._xpoints[i] = 0
            self._ypoints[i] = 0

    def _find_neighbor_point(self, event):
        distance_threshold = 9.0
        nearest_point = None
        min_distance = math.sqrt(2 * (100 ** 2))
        for i in range(3):
            distance = math.hypot(event.xdata - self._xpoints[i],
                                  event.ydata - self._ypoints[i])
            if distance < min_distance:
                min_distance = distance
                nearest_point = (self._xpoints[i], self._ypoints[i])
        if min_distance < distance_threshold:
            return nearest_point
        return None

    def _on_click(self, event):
        # left click
        if event.button == 1 and event.inaxes in [self._axes]:
            point = self._find_neighbor_point(event)
            if point:
                self._dragging_point = point
            self._update_plot()

    def _on_release(self, event):
        if event.button == 1 and event.inaxes in [self._axes
                                                  ] and self._dragging_point:
            self._dragging_point = None
            self._update_plot()

    def _on_motion(self, event):
        if not self._dragging_point:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._dragging_point = self._add_point(event)
        self._update_plot()


if __name__ == "__main__":
    plot = DraggablePlotExample()
