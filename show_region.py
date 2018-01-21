import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

plt.ion()

input_images_names = os.listdir("test_images_corner/.")

for input_image_name in input_images_names:
    if input_image_name.endswith("jpg"):

        print('process file %s' % input_image_name)
        #image = mpimg.imread('test.jpg')
        image = mpimg.imread("test_images_corner/" + input_image_name)
        imshape = image.shape

        print(imshape)

        # straight
        #vertices = np.array([[(0,imshape[0]),(460, 330), (510, 330), (imshape[1],imshape[0])]], dtype=np.int32)
        # right corner
        vertices = np.array([[(0,imshape[0]),(700, 400), (700, 400), (imshape[1],imshape[0])]], dtype=np.int32)
        image_mask = region_of_interest(image, vertices)
        plt.imshow(image_mask, cmap='gray')
        # input('press enter...')
