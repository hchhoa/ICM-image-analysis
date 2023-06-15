from scripts.image_analyser import ImageAnalyser
from scripts.utils import plot_image
import numpy as np
from skimage.segmentation import find_boundaries


class SpleenAnalyser(ImageAnalyser):
    """
    A subclass of ImageAnalyser specifically designed for analysing spleen images.

    This class overrides the run_analysis method to provide a specific workflow
    for spleen image analysis. It also adds an extra method for plotting boundaries.

    Args:
        image_path (str): Path to the image file.

    Methods:
        __init__(): Initializes the SpleenAnalyser with the path of the image.

        run_analysis(): Runs the specific workflow for spleen image analysis. This includes loading
                        the image, preprocessing, blurring, segmenting the image based
                        on a global threshold, applying a watershed algorithm, plotting the cell
                        boundaries and printing the number of cells detected.

        plot_boundaries(): Plots the boundaries found in the watershed image.
    """

    def __init__(self, image_path):
        """Initializes the SpleenAnalyser with the path of the image.

        Args:
            image_path (str): Path to the image file.
        """
        super().__init__(image_path)

    def run_analysis(self):
        """Runs the specific workflow for spleen image analysis.

        This includes loading the image, preprocessing (without blurring),
        segmenting the image based on a global threshold, applying a watershed
        algorithm, plotting the cell boundaries and printing the number of cells detected.
        """

        # Load image
        self.load_image()

        # Preprocess image
        preprocessed_image = self.preprocess_image(self.image, blur=False)

        # Segment image
        mask = self.segment_image(
            preprocessed_image[:, :, 0],
            thresh_type="global",
            thresh=0.12,
            plot=False,
            cmap="gray",
            title="Blue cells nuclei mask",
        )

        # Apply watershed algorithm
        watershed_image = self.watershed_algo(
            mask, min_distance=2, min_object_area=5, plot=True
        )

        # Plot the cells boudaries
        self.plot_boundaries(watershed_image)

        # Count number of blue cells nuclei
        num_cells = np.max(watershed_image)
        print(f"Number of blue cell nuclei: {num_cells}")

    @staticmethod
    def plot_boundaries(watershed_image, cmap="gray", title="Cell boundaries"):
        """Plots the boundaries found in the watershed image.

        Args:
            watershed_image (np.ndarray): The image where the boundaries are plotted.
            cmap (str, optional): The colormap used for plotting. Defaults to 'gray'.
            title (str, optional): The title of the plot. Defaults to 'Cell boundaries'.
        """

        # Find boundaries
        boundaries = find_boundaries(watershed_image)
        # Plot boundaries
        plot_image(boundaries, cmap, title)


if __name__ == "__main__":
    spleen_analyser = SpleenAnalyser("data/Spleen_Hoechst_AutoFL_SMA.ome.tif")
    spleen_analyser.run_analysis()
    # Number of blue cell nuclei: 124054

    spleen_analyser.load_image()
    preprocessed_image = spleen_analyser.preprocess_image(
        spleen_analyser.image, blur=False
    )

    mask = spleen_analyser.segment_image(
        preprocessed_image[:, :, 0],
        thresh_type="global",
        thresh=0.12,
        plot=False,
        cmap="gray",
        title="Blue cells nuclei mask",
    )

    watershed_image = spleen_analyser.watershed_algo(
        mask, min_distance=10, min_object_area=10, plot=True
    )
    num_cells = np.max(watershed_image)
    num_cells

    spleen_analyser.plot_boundaries(watershed_image)
