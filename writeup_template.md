## Advanced Lane Finding

### Xiangjun Fan

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/chess1.png "Undistorted"
[image2]: ./output_images/undist.png "Road Transformed"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/warpped.png "Warp Example"
[image5]: ./output_images/curve.png "Fit Visual"
[image6]: ./output_images/result.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how I addressed each one.   

You're reading it!

### Camera Calibration

#### 1. The camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell with function name `load_and_cal_chess_points()` and `cal_undistort()` of the IPython notebook located in `./advanced_lane.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell with function name `color_and_gradient_threshold()`) in `./advanced_lane.ipynb`.  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Provide an example of a transformed image.

The code for my perspective transform includes cell with function name called `warper()` in `./advanced_lane.ipynb`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([
    [685,450],
    [1100,720],
    [200,720],
    [595,450]
])
dst = np.float32([
    [950,0],
    [950,720],
    [350,720],
    [350,0]
])
```

This resulted in the following source and destination points:

|  Source  | Destination |
| :------: | :---------: |
| 685,450  |    950,0    |
| 1100,720 |   950,720   |
| 200,720  |   350,720   |
| 595,450  |    350,0    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identify lane-line pixels and fit their positions with a polynomial

Then I use histogram and slide window to identify lane lines. As in code cell with function name `hist_and_slide_window()` and `fit_curve()` in `./advanced_lane.ipynb`. The lane lines are fit with a 2nd order polynomial like this:

![alt text][image5]

#### 5. Calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cell with function name `plot_curve()` and `cal_curverad()` in `./advanced_lane.ipynb`

#### 6. Result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in code cell with function name `draw_curve_on_image()` in  `./advanced_lane.ipynb`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to the final video output. 

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took. The techniques I used are all based on the lecture. 

* The camera calibration and distortion correction worked well. 
* To create binary image using color transform and gradients worked well for most of the cases, but for some special case, i.e. low/high light on the road, road with lines other than lane lines, the binary generation may fail. I tried to adjust color and gradient threshold and use h or l channel other than s channel, the code has the optimal threshold in all my trials. 
* Perspective transform worked well too and it is straight forward. However, the transform points are hard coded in the program, and the transformation should be considered to real world dimension calculation of curve and vehicle position. Both can be improved.
* I used histogram and slide window techniques in the lecture to find lane lines, and they worked well. The efficiency can be improved by applying tracking line technique to reduce searching spacing for sliding window.
*  The overall project design is very well and I wished I could have more time to play with those techniques. I learned a ton of computer vision techniques for lane finding in addition to the very first project of basic lane finding. Thanks for preparing such a great project.