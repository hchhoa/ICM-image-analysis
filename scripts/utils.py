import matplotlib.pyplot as plt


def plot_image(image, cmap="gray", title="Image"):
    """
    Plots and displays an image.

    Args:
        image (numpy.ndarray): Image data to be plotted.
        cmap (str): Colormap to be used for visualization (default: 'gray').
        title (str): Title for the plot (default: 'Image').

    """
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()
