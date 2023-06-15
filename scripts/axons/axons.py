from scripts.image_analyser import ImageAnalyser
from scripts.utils import plot_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.morphology import disk, dilation


class AxonsAnalyser(ImageAnalyser):
    """
    A subclass of ImageAnalyser specifically designed for analysing axon images.

    This class provides extra methods for axon analysis, including dilating the axons,
    removing non-circular regions, measuring objects and visualizing measurements.

    Args:
        image_path (str): Path to the image file.

    Methods:
        __init__(): Initializes the AxonsAnalyser with the path of the image.

        dilate_axons():Dilates the axon labels to estimate the axon using disk shaped structural element.

        remove_non_circular_regions():Removes regions from the image that are not circular based on a given threshold.

        measure_objects(): Measures the objects in the segmented, labeled image and returns the diameters and properties.

        run_analysis(): Runs the specific workflow for axon image analysis.
    """

    def __init__(self, image_path):
        super().__init__(image_path)

    def dilate_axons(self, watershed_axons, disk_size=10, plot=False, **kwargs):
        # structural element for dilation
        selem = disk(disk_size)
        labeled_myelin = dilation(watershed_axons, selem)
        if plot:
            cmap = kwargs.get("cmap", "nipy_spectral")
            title = kwargs.get("title", "Image After Watershed")
            plot_image(labeled_myelin, cmap, title)
        return labeled_myelin

    def remove_non_circular_regions(self, mask, thresh=0.9):
        # Convert to grayscale and uint8 if not already
        mask = np.uint8(mask)

        # Find all contours
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on circularity
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            roundness = 4 * np.pi * (area / (perimeter * perimeter))
            if roundness > thresh:
                filtered_contours.append(contour)

        # Create an empty mask to draw the filtered contours
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

        return clean_mask

    def measure_objects(self, labeled_objects):
        # measure the objects in the segmented, labeled image
        props = regionprops(labeled_objects)
        diameters = [prop.equivalent_diameter for prop in props]
        return diameters, props

    def run_analysis(self):
        self.load_image()

        preprocessed_image = self.preprocess_image(
            self.image, kernel_size=(21, 21), beta=1, plot=False
        )

        mask = self.segment_image(preprocessed_image, thresh_type="otsu")
        labeled_mask = axons_analyser.clean_label_mask(
            mask, min_object_area=1200, plot=False
        )
        inverted_mask = (labeled_mask == 0) * 1
        labeled_inverted_mask = self.clean_label_mask(
            inverted_mask, min_object_area=2000, plot=False
        )
        cleaned_labeled_mask = (labeled_inverted_mask == 0) * 1
        plot_image(cleaned_labeled_mask, title="Myelin sheaths segmentation")

        cleaned_mask = self.remove_non_circular_regions(cleaned_labeled_mask, 0.2)

        watershed_axons = self.watershed_algo(
            cleaned_mask,
            min_distance=50,
            min_object_area=1200,
            plot=True,
            title="Axons regions",
        )

        # Axons diameters
        axons_diameters, axons_props = self.measure_objects(watershed_axons)
        mean_axon_diameter = np.mean(axons_diameters)
        print(f"Mean axons diameter: {mean_axon_diameter*0.007}")
        # Mean axons diameter: 0.52 micrometers

        # Myelins diameters
        watershed_myelins = self.dilate_axons(
            watershed_axons,
            disk_size=18,
            plot=True,
            title="Axons + Myelin sheaths regions",
        )
        myelins_diameters, myelins_props = self.measure_objects(watershed_myelins)
        mean_myelin_diameter = np.mean(myelins_diameters)
        print(f"Mean myelins diameter: {mean_myelin_diameter*0.007}")
        # Mean myelins diameter: 0.77 micrometers


if __name__ == "__main__":
    axons_analyser = AxonsAnalyser("data/axons.tif")
    axons_analyser.run_analysis()

    # axons_analyser.load_image()

    # preprocessed_image = axons_analyser.preprocess_image(
    #     axons_analyser.image, kernel_size=(21, 21), beta=1, plot=False
    # )

    # mask = axons_analyser.segment_image(
    #     preprocessed_image, thresh_type="otsu", plot=True
    # )
    # labeled_mask = axons_analyser.clean_label_mask(
    #     mask, min_object_area=1200, plot=True
    # )
    # inverted_mask = (labeled_mask == 0) * 1
    # labeled_inverted_mask = axons_analyser.clean_label_mask(
    #     inverted_mask, min_object_area=2000, plot=False
    # )
    # cleaned_labeled_mask = (labeled_inverted_mask == 0) * 1
    # plot_image(cleaned_labeled_mask)

    # cleaned_mask = axons_analyser.remove_non_circular_regions(cleaned_labeled_mask, 0.2)

    # watershed_axons = axons_analyser.watershed_algo(
    #     cleaned_mask, min_distance=50, min_object_area=1200, plot=True
    # )

    # # Axons diameters
    # axons_diameters, axons_props = axons_analyser.measure_objects(watershed_axons)
    # mean_axon_diameter = np.mean(axons_diameters)
    # print(f"Mean axons diameter: {mean_axon_diameter*0.007}")
    # # Mean axons diameter: 74.36

    # # Myelins diameters
    # watershed_myelins = axons_analyser.dilate_axons(
    #     watershed_axons, disk_size=16, plot=True
    # )
    # myelins_diameters, myelins_props = axons_analyser.measure_objects(watershed_myelins)
    # mean_myelin_diameter = np.mean(myelins_diameters)
    # print(f"Mean myelins diameter: {mean_myelin_diameter*0.007}")
    # # Mean myelins diameter: 109.41
