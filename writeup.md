# **Finding Lane Lines on the Road** 

## Writeup Template

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


---
[This is a link to notebook of this exercise saved as html](P1.html)
---

[//]: # (Image References)

[grayscale]: ./examples/grayscale.jpg "Grayscale"
[canny]: ./examples_pipeline/canny.jpg "Canny"
[masked]: ./examples_pipeline/mask.jpg "Masked"
[hough]: ./examples_pipeline/hough.jpg "Hough"
[final]: ./examples_pipeline/final.jpg "Final"

## Reflection

### 1. Pipeline.

My pipeline consisted of 5 steps. 

![grayscale][grayscale]
1. Convert to grayscale

![canny][canny]
2. Apply Canny edge detection (based on gradient in the image)

![masked][masked]
3. Unmask parts of the images that are not relevant

![hough][hough]
4. Apply Hough Transform (detect straigt lines)

![final][final]
5. Combine the relevant lines to 1 main lines for the left line and one for the right

For step 1. till. 4. I have used the settings that were found during the lessons. The parameters I have found there, were also very succesful with:

A. the sample images that were in the test_images directory. Please check the notebook (or the test_images_output)

B. the solidWhiteRight.mp4 video

C. the solidYellowLeft.mp4 video

## 2. Draw_lines
In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:
1. Input were the lines found by the Hough Transform
2. Based on the x1, I splited all points into left lanes and right lanes points.
3. For both group of points, I calculated the best 1st degreee/linear function
4. With this function, I calculated the full line from the bottom of the image till appr. halfway the image

### 2. Challenge exercise

When doing the challenge exercise, is wasn't that smooth anymore. The following issues where there.

#### different region/size

This video had a different size and there was also part of the bonnet. This could be solved by updating the region.

#### corner

in this video the road and car were making a corner to the right. This also had impact on the region that had to shift to the right. In a production system, probably the turning angle should be a parameter here.

#### shadow from trees and a viaduct with a different color

This was actually the hardest problem. This could be improved by:
- reducing the masking area
- also creating a mask between the 2 lines (a triangle)

After this, I had a reasonable result, allthough more improvements are required. See next section.


### 3. Identify potential shortcomings with your current pipeline

Potential shortcomings are:
1. Different weather condition would probably not work. Can now be seen with the shadow part. Rain and especially snow would probably not work.
2. When making corners, the current mask will not work
3. When making corners, a straight line will not map
4. Worse line markers than this road could also cause issues.

### 4. Suggest possible improvements to your pipeline

Possible improvements could be:
1. Better tuning of parameters with all kind of weather/road conditions. I would also propose an automated learning mechanisme for the parameters (You know the slope of the line a toptimal condition and could use it optimise it with other conditions)
2. The mask should be flexible and adaptable the at least the steering angle of the car.
3. Current the line is 1st order degree, this could 2nd or higher order degree
4. Currently the line is calculated per picture. This could be changed in something that also take into account the previous values.

## Final videos

1. Solid white right lane

[![Solid white right lane](https://img.youtube.com/vi/lgqsa1rrIg0/0.jpg)](https://www.youtube.com/watch?v=lgqsa1rrIg0)

2. Solid yellow left lane

[![Solid yellow left line](https://img.youtube.com/vi/5oZ21K6bMeQ/0.jpg)](https://www.youtube.com/watch?v=5oZ21K6bMeQ)

3. Challenge

[![Challenge](https://img.youtube.com/vi/OY0xBk-eSqs/0.jpg)](https://www.youtube.com/watch?v=OY0xBk-eSqs)

## Files

The following files have been used for testing:
- extract_video_still.py: extract images from all steps in a pipeline from a mp4
- show_region.py: show a region of an image
- challenge.py: the pipeline script for the challenge video (same as in notebook)

[This is a link to notebook of this exercise saved as html](P1.html)

