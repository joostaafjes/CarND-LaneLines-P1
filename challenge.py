# Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from numpy.polynomial import Polynomial


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


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

def region_of_interest2(img, vertices):
    cv2.fillPoly(img, vertices, [0, 0, 0])
    return img

def draw_lines_orig(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if x1 > x_mid and slope > 0.5:
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)
            elif x1 < x_mid and slope < -0.5:
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)

    imshape = img.shape
    max_y = imshape[0]
    max_x = imshape[1]
    left_a = 0
    right_a= 0
    if len(left_x) > 0 and len(left_y) > 0:
        left_a, left_b = np.polyfit(left_x, left_y, 1)
        left_y1 = max_y
        left_x1 = int((left_y1 - left_b) / left_a)
        left_y2 = top_y
        left_x2 = int((left_y2 - left_b) / left_a)
        print(left_x1, left_y1, left_x2, left_y2)
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)

    if len(right_x) > 0 and len(right_y) > 0:
        right_a, right_b = np.polyfit(right_x, right_y, 1)
        right_y1 = max_y
        right_x1 = int((right_y1 - right_b) / right_a)
        right_y2 = top_y
        right_x2 = int((right_y2 - right_b) / right_a)
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)

    return left_a, right_a


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    left_a, right_a = draw_lines(line_img, lines)
    return left_a, right_a, line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image, time=0, save=False):
    # Grayscale
    image_grayscale = grayscale(image)
    plt.imshow(image_grayscale, cmap='gray')
    plt.title('Grayscale')
    plt.show()
    if save:
#        mpimg.imsave('corner_%0.2f.jpg' % time, image_grayscale, cmap='gray')
        mpimg.imsave('test_images_output_video/gray/' + input_file_name + "_t%0.2f.jpg" % (time), image_grayscale, cmap='gray')

    # Gaussian smoothing /blurring
    kernel_size = 7
    image_gaussian_blurring = gaussian_blur(image_grayscale, kernel_size)
    plt.imshow(image_gaussian_blurring, cmap='gray')
    plt.title('Gaussian blurring (%d)' % kernel_size)

    # Canny
    low_threshold = 40
    high_threshold = 120
    image_canny = canny(image_gaussian_blurring, low_threshold, high_threshold)
    plt.imshow(image_canny, cmap='gray')
    plt.title('Canny (low %d and high %d)' % (low_threshold, high_threshold))
    plt.show()
    if save:
        mpimg.imsave('test_images_output_video/canny/' + input_file_name + "_t%0.2f.jpg" % (time), image_canny)

    # Apply mask to relevant part of image
    imshape = image_gaussian_blurring.shape
    max_y = imshape[0] - lower_margin
    max_x = imshape[1]
    vertices = np.array([[(x_margin_lower_left, max_y), (top_x_left, top_y), (top_x_right, top_y), (max_x - x_margin_lower_right, max_y)]], dtype=np.int32)
    image_mask = region_of_interest(image_canny, vertices)
    plt.imshow(image_mask, cmap='gray')
    plt.title('Masked unrelevant parts')
    plt.show()
    if save:
        mpimg.imsave('test_images_output_video/mask/' + input_file_name + "_t%0.2f.jpg" % (time), image_mask)

    # Apply mask 2 to relevant part of image
    vertices2 = np.array([[(mask2_left, max_y), (mask2_center, mask2_top), (mask2_right, max_y)]], dtype=np.int32)
    image_mask2 = region_of_interest2(image_mask, vertices2)
    plt.imshow(image_mask2, cmap='gray')
    plt.title('Masked unrelevant parts 2')
    plt.show()
    if save:
        mpimg.imsave('test_images_output_video/mask2/' + input_file_name + "_t%0.2f.jpg" % (time), image_mask)

    # Hough transformation
    rho = 2  # 2 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # 40 # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    left_a, right_a, image_after_hough = hough_lines(image_mask2, rho, theta, threshold, min_line_length, max_line_gap)
    plt.imshow(image_after_hough, cmap='gray')
    plt.title('Hough')
    plt.show()
    if save:
        mpimg.imsave('test_images_output_video/hough/' + input_file_name + "_t%0.2f_l%0.1f_r%0.1f.jpg" % (time, left_a, right_a), image_after_hough)

    # Weight line
    image_weighted = weighted_img(image_after_hough, image)
    # print("t:%0.2f" % (time))
    if not np.isclose(left_slope, left_a, atol=0.2) or not np.isclose(right_slope, right_a, atol=0.2):
        plt.imshow(image_mask, cmap='gray')
        if save:
            mpimg.imsave('test_images_output_video/weighted/' + input_file_name + "_t%0.2f_l%0.1f_r%0.1f.jpg" % (time, left_a, right_a), image_weighted)
        #print("t:%0.2f_l:%+0.1f_r:%+0.1f" % (time, left_a, right_a))
        plt.show()

    return image_weighted

input_file_name = 'challenge'
time_start = 0
time_end = 10

left_slope = 0.6
right_slope = -0.7

plt.ion()

white_output = 'test_videos_output/' + input_file_name + "_%0.2f_%0.2f.mp4" % (time_start, time_end)
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/" + input_file_name + ".mp4").subclip(time_start, time_end)

top_y = 450
top_x_left = 700
top_x_right = 700
x_mid = 700
x_margin_lower_left = 200
x_margin_lower_right = 150
lower_margin = 60
mask2_left = 300
mask2_center = 650
mask2_right = 900
mask2_top = 450

#for time, frame in clip1.iter_frames(with_times=True):
#    frame_shape = frame.shape
#    process_image(frame, time, save=True)

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

top_y = 330
top_x_left = 460
top_x_right = 510
x_mid = 500
time_start = 0
time_end = 9
x_margin_lower_left = 0
x_margin_lower_right = 0
lower_margin = 0

mask2_left = 300
mask2_center = 500
mask2_right = 700
mask2_top = 400
input_file_name = 'solidWhiteRight'
clip1 = VideoFileClip("test_videos/" + input_file_name + ".mp4").subclip(time_start, time_end)
white_output = 'test_videos_output/' + input_file_name + "_%0.2f_%0.2f.mp4" % (time_start, time_end)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)


time_end = 27
input_file_name = 'solidYellowLeft'
clip1 = VideoFileClip("test_videos/" + input_file_name + ".mp4").subclip(time_start, time_end)
white_output = 'test_videos_output/' + input_file_name + "_%0.2f_%0.2f.mp4" % (time_start, time_end)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
