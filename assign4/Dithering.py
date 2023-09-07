import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load the Samantha image (replace with the actual image file)
image = plt.imread('Samantha.jpg')  # Replace with your image path

# Create the figure and axes for the plot
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Display the initial grayscale image
image_plot = ax.imshow(image, cmap='gray', vmin=0, vmax=255)

# Create the slider axis
ax_variance = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')

# Create the slider
slider_variance = Slider(ax_variance, 'Noise Variance', valmin=1, valmax=21, valstep=2, valinit=1)

# Update function for the slider
def update(val):
    variance = slider_variance.val
    noise = np.random.normal(0, variance, size=image.shape)
    noisy_image = image + noise
    dithered_image = np.where(noisy_image >= 127, 255, 0)
    image_plot.set_data(dithered_image)
    plt.draw()

# Attach the update function to the slider
slider_variance.on_changed(update)

# Set the title
ax.set_title('1-Bit Dithering with Noise')

# Show the plot
plt.show()
