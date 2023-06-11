import tifffile
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import disk, dilation


class AxonsAnalyser:
    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self, plot=False):
        self.tiff = tifffile.TiffFile(self.image_path)
        self.image = tifffile.imread(self.image_path)

        if plot:
            plt.figure()
            plt.imshow(self.image, cmap="gray")
            plt.title("Original Image")
            plt.show()

    def preprocess_image(self, plot=False, **kwargs):
        # Noise reduction
        kernel_size = kwargs.get("kernel_size", (5, 5))
        sigma = kwargs.get("sigma", 0)
        self._blur_image(kernel_size, sigma)

        # Normalisation
        alpha = kwargs.get("alpha", 0)
        beta = kwargs.get("beta", 1)
        norm_type = kwargs.get("norm_type", cv2.NORM_MINMAX)
        dtype = kwargs.get("dtype", cv2.CV_32F)
        self._normalize_image(alpha, beta, norm_type, dtype)

        if plot:
            plt.figure()
            plt.imshow(self.blurred_image, cmap="gray")
            plt.title("Blurred Image")
            plt.figure()
            plt.imshow(self.normalized_image, cmap="gray")
            plt.title("Normalized Image")
            plt.show()

    # Global thresholding
    def threshold_image(self, thresh=0.4, maxval=1, thresh_type="global", plot=False):
        if thresh_type == "global":
            _, self.binary_image = cv2.threshold(
                self.normalized_image, thresh, maxval, cv2.THRESH_BINARY
            )
        else:
            _, self.binary_image = cv2.threshold(
                self.normalized_image, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        if plot:
            # Visualize binary image
            plt.figure(figsize=(10, 10))
            plt.imshow(self.binary_image, cmap="gray")
            plt.title("Binary Image After Otsu Thresholding")
            plt.show()

    def segment_axon(self, min_pixel_size, plot=False):
        labeled_axon = self._label_region(min_pixel_size)
        # distance transform for watershed
        distance = ndi.distance_transform_edt(labeled_axon)
        coords = peak_local_max(distance, min_distance=20, labels=labeled_axon)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labeled_axon = watershed(-distance, markers, mask=labeled_axon)

        if plot:
            # Visualize image after watershed
            plt.figure(figsize=(10, 10))
            plt.imshow(labeled_axon, cmap="nipy_spectral")
            plt.title("Image After Watershed")
            plt.show()

        return labeled_axon

    def measure_objects(self, labeled_objects):
        # measure the objects in the segmented, labeled image
        props = regionprops(labeled_objects)

        diameters = [prop.equivalent_diameter for prop in props]
        return diameters, props

    def visualize_measurements(self, labeled_objects, object_props, object_diameters):
        # Visualize image with diameters
        plt.figure(figsize=(10, 10))
        plt.imshow(labeled_objects, cmap="nipy_spectral")
        plt.title("Image with Measurements")

        for i, prop in enumerate(object_props):
            plt.annotate(
                f"{object_diameters[i]:.2f}",
                (prop.centroid[1], prop.centroid[0]),
                color="white",
            )

        plt.show()

    # Noise reduction
    def _blur_image(self, kernel_size=(5, 5), sigma=0):
        self.blurred_image = cv2.GaussianBlur(self.image, kernel_size, sigma)

    # Normalization
    def _normalize_image(
        self, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ):
        self.normalized_image = cv2.normalize(
            self.blurred_image, None, alpha, beta, norm_type, dtype
        )

    def _label_region(self, min_pixel_size):
        # label the image
        label_image = label(self.binary_image)
        # filter out small objects, assuming they're not axon
        labeled_axon = label_image.copy().astype(np.int32)
        for region in regionprops(label_image):
            if region.area < min_pixel_size:
                labeled_axon[labeled_axon == region.label] = 0
        return labeled_axon


if __name__ == "__main__":
    axons_analyser = AxonsAnalyser("data/axons.tif")
    axons_analyser.load_image()
    axons_analyser.preprocess_image(plot=False, kernel_size=(31, 31), beta=255)
    axons_analyser.threshold_image(thresh=130, plot=False, thresh_type="global")
    labeled_axon = axons_analyser.segment_axon(min_pixel_size=200, plot=True)

    # dilate the axon labels to estimate the axon
    selem = disk(10)  # structural element for dilation, adjust size as needed
    dilated_axon = dilation(labeled_axon, selem)

    # measure diameters
    axon_diameters, axon_props = axons_analyser.measure_objects(labeled_axon)
    mean_axon_diameter = np.mean(axon_diameters)
    print(mean_axon_diameter)
    axons_analyser.visualize_measurements(labeled_axon, axon_props, axon_diameters)

    myelin_diameters, myeline_props = axons_analyser.measure_objects(dilated_axon)
    mean_myelin_diameter = np.mean(myelin_diameters)
    print(mean_myelin_diameter)
    axons_analyser.visualize_measurements(dilated_axon, myeline_props, myelin_diameters)
