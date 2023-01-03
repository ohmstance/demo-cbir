import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def image_show(images, titles="", labels=""):
    """
    Plot images as subplots in a single plot.
    """
    images = images if isinstance(images, list) else [images]
    titles = titles if isinstance(titles, list) else [titles] * len(images)
    labels = labels if isinstance(labels, list) else [labels] * len(images)

    # Num of rows and cols based on num of images
    ncols = min(len(images), 5) # Max 5 imgs per row
    nrows = -(-len(images)//ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(8, 1.8*nrows))
    fig.tight_layout()
    
    # For single subplot, axs is not in np.ndarray -- make sure it is
    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    
    # Show image in each subplot along with title and label
    for i, (image, ax, title, label) in enumerate(
        zip(images, axs.ravel(), titles, labels)
    ):
        # If binary image, use color map 'gray'
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        # If RGB image, make sure color channels are in the third dimension
        elif len(image.shape) == 3:
            if image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            ax.imshow(image)
        ax.set_title(title)
        ax.set_xlabel(label, labelpad=0.0)

        # Hide ticks and numbers for x-axis and y-axis
        ax.tick_params(left=False, bottom=False)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        
    # Delete empty subplots    
    for i in range(i+1, ncols*nrows):
        fig.delaxes(axs.ravel()[i])
    
    # Show figure
    plt.show()