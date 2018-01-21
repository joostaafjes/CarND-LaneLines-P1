# **Finding Lane Lines on the Road** 

## Writeup Template

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

## Reflection

### 1. Pipeline.

My pipeline consisted of 5 steps. 
1. Convert to grayscale
2. Apply Canny edge detection (based on gradient in the image)
3. Unmask parts of the images that are not relevant
4. Apply Hough Transform (detect straigt lines)
5. Combine the relevant lines to 1 main lines for the left line and one for the right

For step 1. till. 4. I have used the settings that were found during the lessons. The parameters I have found there, were also very succesful with:
a. the sample images that were in the test_images directory. Please check the notebook (or the test_images_output)
b. the solidWhiteRight.mp4 video
c. the solidYellowLeft.mp4 video

## 2. Draw_lines
In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:
1. Input were the lines found by the Hough Transform
2. Based on the x1, I splited all points into left lanes and right lanes points.
3. For both group of points, I calculated the best 1st degreee/linear function
4. With this function, I calculated the full line from the bottom of the image till appr. halfway the image

### 2. Challenge exercise

Challenge options:
- part of front ... is visible
- corners
- so mask is working now, later on 2nd degree function in stead of linear

between 4 and 6 seconds
- shadow of trees
- viaduct with more light
If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...


...
Started with the parameters that found during the previous exercises. They weren't actually that good, especially a lot of noise where the lines crossed the horizon.
1. First try: create a better fitting mask. That worked, probably is this area is not easy to get right with this technic.
2. Video white right line is working fine: 
3. Video left yellow has some artifacts, so make tool to extract images that have problems (extract_video_still) 
4. issue that Hough lines always goes from left to right (ordered)


Files:

Notes:
- Notebook not so good with global vars
