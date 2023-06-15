from scripts.utils import plot_image
from abc import ABC, abstractmethod
from pathlib import Path
import tifffile
import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from PIL import Image

OPENSLIDE_PATH = r"D:\Programmes\openslide-win64-20230414\openslide-win64-20230414\bin"

import os

if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import OpenSlide


class FiletypeError(ValueError):
    """An exception raised when an unsupported file type is provided.

    This class is a subclass of ValueError.

    Args:
        message (str): An error message indicating that the provided file type is unsupported.

    Attributes:
        message (str): An error message indicating that the provided file type is unsupported.
    """

    pass


class ImageAnalyser(ABC):
    """
    An abstract base class for image analysis.
    It includes methods for loading, preprocessing (blurring and normalizing)
    TIFF and SVS images, and an abstract method for segmentation.

    Args:
        image_path (str): Path to the image file.

    Attributes:
        image_path (str): Path to the image file.
        image (np.ndarray): Image loaded from the file.

    Methods:
        __init__(): Initializes the ImageAnalyser with the path of the image.

        load_image(): Loads an image from the specified path and checks its dimensions.

        preprocess_image(): Preprocesses an image by applying blurring and normalization.

        segment_image(): Segments an image based on the specified threshold type.

        watershed_algo() : Applies a watershed algorithm to a binary mask.

        clean_label_mask(): Cleans a labeled mask by removing regions smaller than a specified area.

        run_analysis(): Abstract method to run the image analysis.

        _blur_image(): Applies Gaussian blur to an image.

        _normalize_image(): Normalizes an image.

        _check_image_dim(): Checks the dimensions of an image and rearranges if necessary.

    Raises:
        FiletypeError: If the file type is not supported (only TIFF and SVS files).

    """

    def __init__(self, image_path):
        """Initializes the ImageAnalyser class with the path of the image.

        Args:
            image_path (str): Path to the image file.
        """

        self.image_path = image_path

    def load_image(self):
        """Loads an image from the specified path and checks its dimensions.

        This function supports only TIFF and SVS file formats.

        Raises:
            FiletypeError: If the file type is not TIFF or SVS.
        """

        if Path(self.image_path).suffix.lower() == ".tif":
            self.image = tifffile.imread(self.image_path)
            self._check_image_dim()
        elif Path(self.image_path).suffix.lower() == ".svs":
            self.slide = OpenSlide(self.image_path)
        else:
            raise FiletypeError(
                "Invalid file type. Only TIFF and SVS files are supported."
            )

    def preprocess_image(
        self,
        image,
        blur=True,
        kernel_size=(5, 5),
        sigma=0,
        normalize=True,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
        plot=False,
        **kwargs
    ):
        """Preprocesses an image by applying blurring and normalization.

        Args:
            image (np.array): Input image array to be preprocessed.
            blur (bool, optional): Whether to blur the image. Defaults to True.
            kernel_size (tuple, optional): Size of the Gaussian blur kernel. Defaults to (5, 5).
            sigma (int, optional): Standard deviation of the Gaussian blur kernel. Defaults to 0.
            normalize (bool, optional): Whether to normalize the image. Defaults to True.
            alpha (int, optional): Normalizes to range [alpha, beta]. Defaults to 0.
            beta (int, optional): Normalizes to range [alpha, beta]. Defaults to 1.
            norm_type (int, optional): Normalization type. Defaults to cv2.NORM_MINMAX.
            dtype (int, optional): Desired depth of the output image. Defaults to cv2.CV_32F.
            plot (bool, optional): Whether to plot the preprocessed image. Defaults to False.
            **kwargs: Additional parameters for plotting (cmap and title).

        Returns:
            preprocessed_image (np.array): The preprocessed image array.
        """

        if blur:
            # Noise reduction
            blurred_image = self._blur_image(image, kernel_size, sigma)
        else:
            blurred_image = image
        if normalize:
            # Normalisation
            preprocessed_image = self._normalize_image(
                blurred_image, alpha, beta, norm_type, dtype
            )
        else:
            preprocessed_image = blurred_image

        if plot:
            cmap = kwargs.get("cmap", "gray")
            title = kwargs.get("title", "Preprocessed image")
            plot_image(preprocessed_image, cmap, title)

        return preprocessed_image

    def segment_image(
        self,
        image,
        thresh_type="global",
        thresh=0.4,
        block_size=11,
        C=2,
        plot=False,
        **kwargs
    ):
        """Segments an image based on the specified threshold type.

        Args:
            image (np.array): Input image array to be segmented.
            thresh_type (str, optional): Type of thresholding. Options are "global", "global_inv", "adaptive", "otsu". Defaults to "global".
            thresh (float, optional): Threshold value for "global" and "global_inv" methods. Defaults to 0.4.
            block_size (int, optional): Size of a pixel neighborhood for "adaptive" method. Defaults to 11.
            C (int, optional): Constant subtracted from the mean for "adaptive" method. Defaults to 2.
            plot (bool, optional): Whether to plot the segmented image. Defaults to False.
            **kwargs: Additional parameters for plotting (cmap and title).

        Returns:
            mask (np.array): The binary mask of the segmented image.
        """

        if thresh_type == "global":
            _, mask = cv2.threshold(image, thresh, 1, cv2.THRESH_BINARY)
        elif thresh_type == "global_inv":
            _, mask = cv2.threshold(image, thresh, 1, cv2.THRESH_BINARY_INV)
        elif thresh_type == "adaptive":
            image_adaptive = cv2.convertScaleAbs(image * 255)
            mask = cv2.adaptiveThreshold(
                image_adaptive,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                C,
            )
        elif thresh_type == "otsu":
            image_otsu = cv2.convertScaleAbs(image * 255)
            _, mask = cv2.threshold(
                image_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        if plot:
            cmap = kwargs.get("cmap", "gray")
            title = kwargs.get("title", "Segmented image")
            plot_image(mask, cmap, title)
        return mask

    def watershed_algo(
        self, mask, min_distance, min_object_area=50, plot=False, **kwargs
    ):
        """Applies a watershed algorithm to a binary mask.

        Args:
            mask (np.array): Input binary mask for watershed algorithm.
            min_distance (int): Minimum distance between peaks in the distance map.
            min_object_area (int, optional): Minimum area of objects to be retained. Defaults to 50.
            plot (bool, optional): Whether to plot the image after applying watershed algorithm. Defaults to False.
            **kwargs: Additional parameters for plotting (cmap and title).

        Returns:
            watershed_image (np.array): The image after applying the watershed algorithm.
        """

        # label image
        clean_labeled_mask = self.clean_label_mask(mask, min_object_area)

        # distance transform for watershed
        distance = ndi.distance_transform_edt(clean_labeled_mask)
        coords = peak_local_max(distance, min_distance, labels=clean_labeled_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        watershed_image = watershed(-distance, markers, mask=clean_labeled_mask)

        if plot:
            cmap = kwargs.get("cmap", "nipy_spectral")
            title = kwargs.get("title", "Image After Watershed")
            plot_image(watershed_image, cmap, title)

        return watershed_image

    @staticmethod
    def clean_label_mask(mask, min_object_area=50, plot=False, **kwargs):
        """Cleans a labeled mask by removing regions smaller than a specified area.

        Args:
            mask (np.array): Input labeled mask to be cleaned.
            min_object_area (int, optional): Minimum area of objects to be retained. Defaults to 50.
            plot (bool, optional): Whether to plot the cleaned image. Defaults to False.
            **kwargs: Additional parameters for plotting (cmap and title).

        Returns:
            cleaned_mask (np.array): The cleaned labeled mask.
        """

        labeled_mask = label(mask)
        cleaned_mask = labeled_mask.copy().astype(np.int32)
        for region in regionprops(labeled_mask):
            if region.area < min_object_area:
                cleaned_mask[cleaned_mask == region.label] = 0
        if plot:
            cmap = kwargs.get("cmap", "gray")
            title = kwargs.get("title", "Image after removing small regions")
            plot_image(cleaned_mask, cmap, title)
        return cleaned_mask

    @abstractmethod
    def run_analysis(self):
        """Abstract method to run the image analysis. To be implemented in subclasses."""
        pass

    @staticmethod
    def _blur_image(image, kernel_size=(5, 5), sigma=0):
        """Applies Gaussian blur to an image.

        Args:
            image (np.array): Input image array to be blurred.
            kernel_size (tuple, optional): Size of the Gaussian blur kernel. Defaults to (5, 5).
            sigma (int, optional): Standard deviation of the Gaussian blur kernel. Defaults to 0.

        Returns:
            blurred_image (np.array): The blurred image array.
        """

        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
        return blurred_image

    # Normalization
    @staticmethod
    def _normalize_image(
        image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ):
        """Normalizes an image.

        Args:
            image (np.array): Input image array to be normalized.
            alpha (int, optional): Normalizes to range [alpha, beta]. Defaults to 0.
            beta (int, optional): Normalizes to range [alpha, beta]. Defaults to 1.
            norm_type (int, optional): Normalization type. Defaults to cv2.NORM_MINMAX.
            dtype (int, optional): Desired depth of the output image. Defaults to cv2.CV_32F.

        Returns:
            normalized_image (np.array): The normalized image array.
        """

        normalized_image = cv2.normalize(image, None, alpha, beta, norm_type, dtype)
        return normalized_image

    # Check image dimensions
    def _check_image_dim(self):
        """Checks the dimensions of an image and rearranges if necessary.

        If the image is grayscale, an extra dimension is added.
        If the image is in the wrong orientation, it is transposed.
        """

        # Check the number of channels and rearrange dimensions if necessary
        if len(self.image.shape) == 2:
            self.image = np.expand_dims(self.image, axis=2)
        elif len(self.image.shape) == 3 and self.image.shape[0] < self.image.shape[2]:
            self.image = np.transpose(self.image, axes=(1, 2, 0))
