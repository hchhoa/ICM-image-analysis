from scripts.image_analyser import ImageAnalyser
from scripts.utils import plot_image
from PIL import Image
import cv2
import numpy as np
from skimage.filters import threshold_otsu
import os
import matplotlib.pyplot as plt

OPENSLIDE_PATH = r"D:\Programmes\openslide-win64-20230414\openslide-win64-20230414\bin"

if hasattr(os, "add_dll_directory"):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

from openslide import OpenSlide


class LungAnalyser(ImageAnalyser):
    def __init__(self, image_path):
        super().__init__(image_path)

    def get_region(self, x, y, level, width, height):
        """Reads a region from the slide."""
        region = self.slide.read_region((x, y), level, (width, height))
        self.image = np.array(region)[:, :, :3]  # Discard alpha if it exists
        return self.image

    def run_analysis(self):
        pass


# Read the images
control_slide = LungAnalyser("data/Lung_isotypic control.svs")
control_slide.load_image()
marker_slide = LungAnalyser("data/Lung_marker of interest.svs")
marker_slide.load_image()

control_slide.slide.properties
# 23411x41431
control_slide.get_region(23400, 41100, 0, 10, 10)
control_slide.image.shape
control_slide.to_grayscale()

mask = ImageAnalyser.segment_image(
    image=control_slide.image_gray,
    thresh_type="adaptive",
    block_size=11,
    C=2,
    plot=True,
)

plt.imshow(control_slide.image_binary, cmap="gray")
plt.show()

marker_slide.get_region(2500, 12000, 0, 10000, 5000)
marker_slide.image.shape
plt.imshow(marker_slide.image, cmap="gray")
plt.show()
