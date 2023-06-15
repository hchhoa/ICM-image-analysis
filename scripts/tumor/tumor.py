from scripts.image_analyser import ImageAnalyser
from scripts.utils import plot_image
from skimage import morphology
import numpy as np
from skimage.segmentation import find_boundaries
from collections import Counter
from scipy.ndimage import distance_transform_edt


class TumorAnalyser(ImageAnalyser):
    """
    A subclass of ImageAnalyser specifically designed for analysing tumor images.

    This class provides additional methods for tumor analysis, including attributing phenotypes,
    counting phenotypes, computing minimal distances, and others.

    Args:
        image_path (str): Path to the image file.

    Methods:
        __init__(): Initializes the TumorAnalyser with the path of the image.

        attribute_phenotypes(): Attributes phenotypes to cells based on the maximum expression marker.

        count_phenotypes(): Counts the occurrences of each phenotype and calculates their proportions.

        compute_mean_minimal_distances(): Computes the mean minimal distances of phenotypes to the stroma.

        run_analysis(): Runs the specific workflow for tumor image analysis.

        combine_masks(): Combines multiple masks into a composite mask.

        analyze_marker_expression(): Analyzes marker expression for cells in a composite mask.

        perform_closing(): Performs a morphological closing operation on a binary mask.

        plot_boundaries(): Plots boundaries in a watershed image.
    """

    def __init__(self, image_path):
        """
        Initializes the TumorAnalyser with the path of the image.

        Args:
            image_path (str): Path to the image file.
        """
        self.image_path = image_path

    def attribute_phenotypes(self, cells_expression):
        """
        Attributes phenotypes to cells based on the maximum expression marker.

        Args:
            cells_expression (List[List[int]]): A nested list containing the expression levels of each cell.
        """

        attributed_phenotypes = []
        phenotypes = [
            "PDL1-expressed phenotype",
            "CD8-expressed phenotype",
            "FoxP3-expressed phenotype",
            "CD68-expressed phenotype",
            "PD1-expressed phenotype",
        ]
        for cell_expression in cells_expression:
            max_index = cell_expression.index(max(cell_expression))

            attributed_phenotypes.append(phenotypes[max_index])

        self.attributed_phenotypes = np.array(attributed_phenotypes)

    def count_phenotypes(self, attributed_phenotypes):
        """
        Counts the occurrences of each phenotype and calculates their proportions.

        Args:
            attributed_phenotypes (np.ndarray): An array of attributed phenotypes.
        """

        # Count the occurrences of each phenotype
        phenotype_counts = Counter(attributed_phenotypes)

        # Calculate the proportion of each phenotype
        total_cells = len(attributed_phenotypes)
        proportions = {
            phenotype: np.round(count / total_cells, 3)
            for phenotype, count in phenotype_counts.items()
        }

        self.proportions = proportions

    def compute_mean_minimal_distances(
        self, stroma_mask, composite_mask, attributed_phenotypes
    ):
        """
        Computes the mean minimal distances of phenotypes to the tumor tissue.

        Args:
            stroma_mask (np.ndarray): A binary mask representing the stroma.
            composite_mask (np.ndarray): A binary mask representing the composite of multiple masks.
            attributed_phenotypes (np.ndarray): An array of attributed phenotypes.
        """
        mean_minimal_distances = []
        distance_map = distance_transform_edt(stroma_mask)

        labeled_cells = composite_mask.copy()
        labels = np.unique(labeled_cells)[1:]

        phenotypes = np.unique(attributed_phenotypes)
        for phenotype in phenotypes:
            print(phenotype)
            min_distances = []
            phenotype_mask = attributed_phenotypes == phenotype
            labels_mask = labels[phenotype_mask]
            for label in labels_mask:
                labeled_cells_mask = labeled_cells == label
                min_distance = np.min(distance_map[labeled_cells_mask])
                min_distances.append(min_distance)

            mean_minimal_distances.append(np.mean(min_distances))

        self.mean_minimal_distances = mean_minimal_distances

    def run_analysis(self):
        """
        Runs the specific workflow for tumor image analysis.
        """
        # Q1
        self.load_image()
        preprocessed_image = self.preprocess_image(
            self.image[:, :, 5], kernel_size=(7, 7), plot=False
        )
        mask = self.segment_image(
            preprocessed_image,
            thresh_type="global",
            thresh=0.06,
            block_size=21,
            C=2,
            plot=False,
        )

        closed_mask = self.perform_closing(mask, disk_size=8, plot=False)
        cleaned_labeled_mask = self.clean_label_mask(
            closed_mask, min_object_area=500, plot=False
        )
        tumor_mask = (cleaned_labeled_mask > 0) * 1
        plot_image(tumor_mask, title="Tumor mask")

        stroma_mask = (tumor_mask == 0) * 1
        plot_image(stroma_mask, title="Stroma mask")

        ### Q2
        inflammatory_cells_masks = []
        for channel in range(5):
            preprocessed_image = self.preprocess_image(
                self.image[:, :, channel] * stroma_mask,
                blur=False,
                plot=False,
            )

            mask = self.segment_image(
                preprocessed_image,
                thresh_type="otsu",
                thresh=0.2,
                block_size=51,
                plot=False,
                cmap="gray",
                title="Inflammatory cells mask",
            )

            watershed_image = self.watershed_algo(
                mask, min_distance=5, min_object_area=20, plot=False
            )
            # Plot the cells boudaries
            self.plot_boundaries(
                watershed_image,
                title=f"Channel {channel} inflammatory cells boundaries",
            )

            # cells_analyser.plot_boundaries(watershed_image)
            inflammatory_cells_masks.append(watershed_image)

        self.inflammatory_cells_masks = inflammatory_cells_masks

        ### Q3

        # Assuming you have individual labeled masks as labeled_masks (a list of NumPy arrays)
        composite_mask = self.combine_masks(inflammatory_cells_masks)

        preprocessed_image = self.preprocess_image(self.image, plot=False)

        cells_expression = self.analyze_marker_expression(
            composite_mask, preprocessed_image
        )

        self.attribute_phenotypes(cells_expression)
        print(self.attributed_phenotypes)

        ### Q4
        # Count the occurrences of each phenotype
        self.count_phenotypes(self.attributed_phenotypes)

        # Print the proportions
        for phenotype, proportion in self.proportions.items():
            print(f"{phenotype}: {proportion}")

        ### Q5
        self.compute_mean_minimal_distances(
            stroma_mask, composite_mask, self.attributed_phenotypes
        )
        print(self.mean_minimal_distances)

    @staticmethod
    def combine_masks(inflammatory_cells_masks, label_offset=2000):
        """
        Combines multiple masks into a composite mask.

        Args:
            inflammatory_cells_masks (List[np.ndarray]): List of masks for each image.
            label_offset (int, optional): A value to offset the labels for each cell. Defaults to 2000.

        Returns:
            np.ndarray: The composite mask.
        """

        # Assuming you have individual labeled masks as labeled_masks (a list of NumPy arrays)
        composite_mask = np.zeros_like(inflammatory_cells_masks[0])

        # Assign a unique label to each cell in the composite mask
        for mask in inflammatory_cells_masks:
            mask = np.where(mask > 0, mask + label_offset, 0)
            composite_mask = np.maximum(composite_mask, mask)
            label_offset += np.max(mask)

        return composite_mask

    @staticmethod
    def analyze_marker_expression(composite_mask, preprocessed_image):
        """
        Analyzes marker expression for cells in a composite mask.

        Args:
            composite_mask (np.ndarray): A binary mask representing the composite of multiple masks.
            preprocessed_image (np.ndarray): The preprocessed image.

        Returns:
            List[List[float]]: A list of lists containing the marker expression for each cell.
        """

        cells_expression = []
        for cell_index in np.unique(composite_mask)[1:]:
            cell_mask = composite_mask == cell_index
            cell_expression = []

            for channel in range(5):
                image_channel = preprocessed_image[:, :, channel]
                channel_mean_intensity = np.mean(image_channel[cell_mask])
                cell_expression.append(channel_mean_intensity)

            cells_expression.append(cell_expression)

        return cells_expression

    @staticmethod
    def perform_closing(mask, disk_size, plot=False, **kwargs):
        """
        Performs a morphological closing operation on a binary mask.

        Args:
            mask (np.ndarray): The binary image mask.
            disk_size (int): The size of the disk for the morphological operation.
            plot (bool, optional): Whether to plot the image after operation. Defaults to False.
            **kwargs: Additional keyword arguments for plot_image function.

        Returns:
            np.ndarray: The resulting mask after morphological closing operation.
        """

        # Define the disk of size `selem_size`.
        disk = morphology.disk(disk_size)

        # Apply the morphological closing operation.
        closed_mask = morphology.closing(mask, disk)

        if plot:
            # Visualize binary image
            cmap = kwargs.get("cmap", "gray")
            title = kwargs.get("title", "Preprocessed image")
            plot_image(closed_mask, cmap, title)
        return closed_mask

    @staticmethod
    def plot_boundaries(watershed_image, cmap="gray", title="Cell boundaries"):
        """
        Plots boundaries in a watershed image.

        Args:
            watershed_image (np.ndarray): The image obtained after the watershed algorithm.
            cmap (str, optional): The color map to use for plotting. Defaults to "gray".
            title (str, optional): The title for the plot. Defaults to "Cell boundaries".
        """
        # Find boundaries
        boundaries = find_boundaries(watershed_image)
        # Plot boundaries
        plot_image(boundaries, cmap, title)


if __name__ == "__main__":
    tumor_analyser = TumorAnalyser("data/Multiplexing image_cancer-inflamation.tif")
    tumor_analyser.run_analysis()

    # PD1-expressed phenotype: 0.484
    # PDL1-expressed phenotype: 0.475
    # CD8-expressed phenotype: 0.033
    # FoxP3-expressed phenotype: 0.002
    # CD68-expressed phenotype: 0.005
    # [45.254915583840365, 30.723161179970745, 49.927262455018855, 38.195564676444, 32.9678816283516]
