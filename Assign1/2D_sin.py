from matplotlib.widgets import Slider
import numpy as np
from matplotlib import pyplot as plt


m, n = np.meshgrid(np.arange(1, 257), np.arange(1, 257))
phi_initial = 0.0

# Define the update function for the slider
def update(val):
    phi = slider_phi.val
    Z = np.cos(2 * np.pi * 0.2 * m + 2 * np.pi * 0.3 * n + phi)
    ax.clear()
    ax.imshow(Z, cmap='viridis')
    plt.draw()

# Create the figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Create the initial plot
Z = np.cos(2 * np.pi * 0.2 * m + 2 * np.pi * 0.3 * n + phi_initial)
im = ax.imshow(Z, cmap='viridis')

# Create the slider axes
ax_phi = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_phi = Slider(ax_phi, 'Phi', 0, 2 * np.pi, valinit=phi_initial)

# Attach the update function to the slider
slider_phi.on_changed(update)

# Show the plot
plt.show()
