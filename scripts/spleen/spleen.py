import cv2
import numpy as np
from skimage.measure import label, regionprops
import tifffile
import matplotlib.pyplot as plt


class CellCounter:
    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self):
        self.image = tifffile.imread(self.image_path)

    def pre_process(self, plot=False):
        self.blurred = np.zeros_like(self.image)
        self.normalized = np.zeros_like(self.image, dtype=np.float32)

        for channel in range(self.image.shape[0]):
            # Noise reduction
            self.blurred[channel] = cv2.GaussianBlur(self.image[channel], (5, 5), 0)

            # Normalization
            self.normalized[channel] = cv2.normalize(
                self.blurred[channel],
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
            if plot:
                plt.figure()
                plt.imshow(self.normalized[channel], cmap="gray")
                plt.title("Normalized Image")
                plt.show()

    def global_thresholding(self, channel, threshold=0.2, plot=False):
        _, self.mask = cv2.threshold(
            self.normalized[channel], threshold, 1, cv2.THRESH_BINARY
        )
        if plot:
            plt.figure()  # Create a new figure
            plt.imshow(self.mask, cmap="gray")
            plt.title("Global Thresholding")
            plt.show()

    def compare_channels(self, plot=False):
        # Create a mask where the blue channel has the highest value
        self.mask = np.logical_and(
            self.normalized[0] > 1.4 * self.normalized[1],
            self.normalized[0] > 1.4 * self.normalized[2],
        )
        self.mask = self.mask.astype(np.uint8)

        if plot:
            plt.figure()  # Create a new figure
            plt.imshow(self.mask, cmap="gray")
            plt.title("Blue Cells Mask")
            plt.show()

    def extract_features(self, plot=False):
        self.label_img = label(self.mask)
        self.regions = regionprops(self.label_img)

        binary_img = self.label_img > 0

        # Visualize labels
        if plot:
            plt.figure()
            plt.imshow(binary_img)
            plt.title("Labelled Regions")
            plt.show()

    def count_cells(self):
        num_cells = len(self.regions)
        return num_cells


if __name__ == "__main__":
    cell_counter = CellCounter("data/Spleen_Hoechst_AutoFL_SMA.ome.tif")
    cell_counter.load_image()
    cell_counter.pre_process()
    cell_counter.compare_channels()
    # cell_counter.global_thresholding(0, 0.25)
    cell_counter.extract_features(plot=False)

    # Count the cells
    num_cells = cell_counter.count_cells()
    print(f"Number of blue cell nuclei: {num_cells}")
