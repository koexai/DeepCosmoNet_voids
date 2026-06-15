"""
this module is meant to plot the training progress live using matplotlib
"""
import os
import matplotlib.pyplot as plt
import time
import random


class LivePlot:
    """
    The class represents a matplotlib live plot for a single metric
    """

    def __init__(self, title="Training Progress"):
        self.train_metric = []
        self.val_metric = []
        # plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots()
        self.title = title
        self.ax.set_title(self.title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel(title)
        (self.train_line,) = self.ax.plot([], [], label="Train", color="b")
        (self.val_line,) = self.ax.plot([], [], label="Val", color="r")
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw()
        # self.fig.show()

    def update(self, dot, mode):
        """
        Updates the plot
        Args:
            dot: the new value to add to the plot
            mode: "train" or "val
        """
        if mode == "train":
            self.train_metric.append(dot)
            self.train_line.set_data(range(len(self.train_metric)), self.train_metric)
        if mode == "val":
            self.val_metric.append(dot)
            self.val_line.set_data(range(len(self.val_metric)), self.val_metric)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, folder):
        """
        Saves the live plot to a folder
        Args:
            folder: path to save the figure
        """
        self.fig.savefig(os.path.join(folder, self.title + ".png"))


def test_live_plot():
    """
    Tests the live plot function.

    Returns:
    - None
    """
    plotter = LivePlot()
    plotter2 = LivePlot()
    for epoch in range(50):
        train_loss = 1 / (epoch + 1) + random.uniform(-0.05, 0.05)
        val_loss = 1 / (epoch + 1) + random.uniform(-0.1, 0.1)
        plotter.update(train_loss, mode="train")
        plotter2.update(1 - train_loss, mode="train")
        plotter.update(val_loss, mode="val")
        plotter2.update(1 - val_loss, mode="val")
        time.sleep(0.1)


if __name__ == "__main__":
    test_live_plot()
