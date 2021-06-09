**Advanced Lane Finding Project**
Ryan O'Shea

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted]: ./output_images/undistorted.jpg "Undistorted"
[distorted]: ./test_images/test1.jpg "Undistorted"
[binary]: ./output_images/binary_scene.jpg "Binary"
[perspective]: ./output_images/warped_lanes.jpg "Binary Example"
[warped_lane_lines]: ./output_images/warped_lane_lines.jpg "Identified lines"
[warped_lane_poly]: ./output_images/warped_lane_lines_poly.png "Fitted polynomial"
[unwarped_lane]: ./output_images/unwarped_lane_lines.jpg "Highlighted lane"
[final_res]: ./output_images/final_result.jpg "Final result"
[final_res_text]: ./output_images/final_result_w_text.jpg "Final result with text"
[video1]: ./output_videos/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Pipeline (single images)

The main pipeline code can be found in the third code block in the Python notebook named lane_finder.ipynb. It takes in an image, camera parameters, distortion matrix, and two line objects and returns an image with the lane identified.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and fifth cells in the Ipython notebook names lane_finder.ipynb. The code, the function named `calibrateCam()` in the second cell is part of the block containing all the helper functions for the code. The code in the fifth block contains the actual call to the function that gets the camera parameter matrix and distortion matrix.

In the function code in block 2 I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  Corner detection was performed using the `cv2.findChessboardCorners()` function. This function took in a grayscale version of the movie and hte chessboard square dimensions. In our case the dimensions were 9x6. The results of the corner finding was validated using the `cv2.drawChessboardCorners()` function. The found corners lined up very well with the image that they were overlayn on. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistorted]

The original image can be seen below. There aren't major difference between the two images which is likely due to the camera that was used to capture the image not having much distortion. If a fish eye camera cheap camera were used then the distortion correction would likely be significantly more noticable.

![alt text][distorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image using the `getBinaryLanes()` function in the second code block of the Python notebook. The function first converts the image into hls color space so the l and s channels can be isolated and used. Opencv's Sobel function is used to a apply a sobel derivative filter across the x dimension of the l channel of the image. After taking the absolute value of the result and scaling it to values 0-255 it is binarized by creating a blank image and then activating the pixels on it where the pixels in the sobel image fall within the designated threshold. The result of this identifies areas with high gradients along the x axis. This corresponds to strong vertical edges which are good representations of lane lines. A color threshold was also aplied to the s channel of the image. This was found to a better identifier of yellow line and shaded area than the sobel method so it  combines well with the output from the Sobel thresholding. These methods were combined by creating a blank activation image and then activating the pixels only where there was an activation in both the sobel and s channel image. An example of the output can be seen in the image below.

![alt text][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `getTransformedLaneView()`, which appears in the second block of the python notebook. The fucntion takes in an image as well the src point which are passed in as a region of interest.  I chose to partially the hardcode the source and destination points in the following manner:

```python
# Boundaries for the ROI. Might need more tuning
bottom_bound = height - 10
top_bound = 450
left_lane_end = 550
right_lane_end = 725
# top left, top right, bottom right, bottom left
ROI = np.float32([[left_lane_end, top_bound], [right_lane_end, top_bound], [width, bottom_bound], [0, bottom_bound]])

dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. It was also easy enough to identify the correctness by eye because the transform should result in a bird's eye view of the lane area. An example of the birds eye view transformation can be seen in the image below.

![alt text][perspective]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify the lane lines and their positions the `fitPolynomial()` and `findInitialLanePixels()`functions in block 2 of the notebook. The `findInitialLanePixels()`function takes care of the actual identification of lane pixels by first creating a histogram along the x axis of the image which highlights the areas of the image where are large amounts of activated pixels with the same x value. These areas correspond to vertical lines and the two largest peaks will be strong candidates for the two lane lines. The left lane line is found using a simple check for the highest histogram peak on the left side of the image with the same method being used for the the right side. The indices of all non zero x and y pixels are then found using the numpoy nonzero() function. N sliding windows are then created for each side of the image which will be used to scan for and find the the average position of groups of lane pixels across the height of the image. The dimensions of the sliding windows are defined and used to determine which pixels fall within in the window of if they are part of the group that makes up the line. If a window is found to encompass a number of pixels above the defined minimum pixel value then its position is updated to be the new position for that windo and the indices of the encompassed pixels are stored in their respective arrays. Once all the left lane and right pixels have been identified by their respective windows they are grouped together and returned to the `fitPolynomial()` function. An example of the identified left and right line pixels and the sliding windows used to identify them can be seen in the image below.

![alt text][warped_lane_lines]

Once the pixels have been found for each of the respective lanes they can now be used to create a line of best fit based on the postion of the pixels. The `fitPolynomial()` function uses the `np.polyfit()' function to find the best second order polynomial that fits the data based on the set of inputs and outputs that we gave it in the form of x, y indice pairs. The calculated polynomials are then plotted over the previous image in order to display how well the equation fits the data in addition to the most likely future course of the lane line given the current data. An examples of this can be seen in the image below.

![alt text][warped_lane_poly]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculate in the function `getLineCurvatures()` in the seconde code block with the rest of the major supporting functions. This function first calculates the meters per pixel scaling factor in both the x and y directions using the predefined values for each which are 3.7 for x and 30 for y. These scaling factors are then used to create new equation of best fit that have been properly scaled to have their dimensions be in meters instead of pixels. The left curvature and right curvature are then calculated using the provided mathematical formula which then codes as shown below:
```python
    left_curverad = ((1 + (2*left_fit_m[0]*y_eval*ym_per_pix + left_fit_m[1])**2)**1.5)/(np.absolute(2*left_fit_m[0]))
    right_curverad = ((1 + (2*right_fit_m[0]*y_eval*ym_per_pix + right_fit_m[1])**2)**1.5)/(np.absolute(2*right_fit_m[0])) 
```

The distance from center was calculated by making use of the given assumption that the camera is mounted in the exact middle of the car so the distance from the center of the image to the center of the lane should be the offset.

This was calculated using the following block of code.

```python
    # Get the average location of the left and right lane
    midpoint = width/2
    
    # Get the subset of line pixels that are closest to the car. In this case we get consider all pixels that are closer than 90% of the
    # y value of the closest pixels
    close_leftx = leftx[lefty > (.9 * np.max(lefty))]
    close_rightx = rightx[righty > (.9 * np.max(righty))]
    
    left_avg = np.mean(close_leftx)
    right_avg = np.mean(close_rightx)
    
    # Calculate distance from center
    lane_center = (left_avg + right_avg)/2
    offset_from_center = abs(midpoint - lane_center)*xm_per_pix
```

Because the distance is initially calculated in pixels it needs to be scaled using the previously calculated scaling factor to obtain the offset in meters. The calculated curvature and offset can be seen in the image below. The curvature radius is uncharacteristically high because this stretch of road happens to have nearly straight lines. In the results video it can be seen that the curvature normally fluctuates between 500 and 1500m which matches very nicely with the 1km estimate given to us at the start of the project. These values were drawn onto the final image using the OpenCV `putText()` function.

![alt text][final_res_text]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lane area was found in the `pipeline()` function in the 3rd major code block of the notebook. The lane area is identified by first transforming the sets of points for the left lane and right lane into a format that `cv2.fillPoly` can use through a seriers of transpose and stacking operations on the respective arrays. The fillPoly function is then used to highlight the area between the two sets of points representing the lane lines which is displayed as a green rectangle that follows the curve of the lane. This image is then warped back into the original view by using the inverse warp matrix that we calculated earlier during the initial warping process and the `cv2.warpPersepective` function. The result of this operation can be seen in the image below.

![alt text][unwarped_lane]

This highlighted lane is then combined with the original undistorted image to create a bright green highlight of the lane area projected into the real world. The `cv2.addWeighted` function was used for this because it combines two images toghether by avering the values in the two image to create an averaged output. The result of this can be seen in the image below.

![alt text][final_res]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Video functionality was added to the main driver code which can be found in the 6th block of the notebook. Video files were opened using OpenCV's `VideoCapture` function and the output video was created using OpenCV's `VideoWriter` function. Frames were read one by one from the video and then fed into the pipeline in order to recieve the fully processed image. The relevant text was then drawn onto the frame and the frame was written to the output file before moving onto the next frame. The video playing and writing process seems to take a fair bit of time which is a major room for improvement in the future. The output from the project video can be seen in the embedded video or the link below. If the videos do not play in the browser then they may need to be downloaded. For some reason even though I was saving them as mp4s the jupyter notebook would not let me play them in my browser. However, when dowloaded to my computer they ran perfectly fine.

![alt text][video1]

Here's a [link to my video result](./output_videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the main results video the pipeline actually does a fairly good job of highlighting the lane throughout most of the video with no major failures throughout the entire video. There is however jumpiness at the very end and beginning of the rectangle in areas where the pavement is very light. In these areas the lane lines are much harder to detect due to the lack of contrast between the lines and the pavement. This is particularly a problem for the yellow line. A number of solutions could be implementd in the future to fix this. The existing method for binarizing the lane lines is effective but could definitely be made more robust. A simple approach would be to try further tuning the threshold values which would likley lead to small increases in performance. A more robust aproach would involve trying to use the h channel to better detect the yellow line in the image. The channel represents the different hues of colors in an image so it should have a range of values that correspond to yellow in the image.

Another major improvemnt to fix the jitter in the lines in lightly colored pavement areas would be to implement the recommendation of averaging the polynomials used to represent the lanes over time. This would prevent one bad detection from drastically altering the direction and shape of the predicted lane. I attempted this using the provided Line class but was unable to accomplish it in time for the project submission due to some persistant errors. My approach was to avergae the most recent 20 polynomials proposed for each of the lanes and store those as a best fit for the lane. If the number of stored lanes passed 20 then the oldest stored fit was removed from the list. The code for this can be found in the `updateLine()` function in the 4th code block and can also be seen below.

```python
    # Append latest fit to list of recents
    self.recent_xfitted.append(self.current_fit)
    
    # If there are now more than 20 fits pop off the oldest one
    if len(self.recent_xfitted) > 20:
        self.recent_xfitted.pop(0)
    
    # Take the mean of the last 20 fits to get the average best fit
    self.best_fit = np.mean(self.recent_xfitted, axis=0)
```

If I had gotten this to work it should have produced a much smoother line with less jitter in problematic areas in the vidoe. The number of saved fit parameters would of course need to be tuned to achieve the best results.

In order to make use of this best fit line the `searchAroundPoly()` and `fit_poly()` functions were taken from the previous lessons and built upon to accomodate having Line objects passed in. The `searchAroundPoly()` function takes the existing polynomial and searchs around for lane points around it within a defined margin in order to perform a much more efficient and targeted search. The newly found line would be added to the existing running average for each lane object and the new best fit would be caluclated and used to find the lane area.

The pipeline performs poorly on the challenge videos for the project due to some of the reasons listed above as well as others that are more dificult to fix. The one video shows a very stong line in the middle of the lane due to poor road maintenance which gets picked up as a stronger line than the left yellow line. A simple approach to fix this would be to ignore lines that appear in the area of the image that should correspond with the area betweeen the two lanes. This would unfortunately cause problems on complex roads but it would likely be a good place to start. The previously mentioned improvements to lane line isolation through the use of the h channel may also improve performance here. The harder challenge video has a wide array of challenges that would need to be addressed. One of the main ones is the highly dynamic lighting of the environment that occasionally totally blocks the cameras view of the lanes. This would make detection nearly impossible and leads to very poor results. Once solution for this would be to add a predictive functionality to the lane lines that can predict the future line appearance based on current and past data. This could potentially be implemented through the use of a Kalman filter which is great for creating models that are robust to rapid noisy change in the environment and are capable of predicting future states of a system based on the current state. The complex and curving nature of the road in the harder video would also likely require complete defintion of how the ROI and general perspective transform process is performed.
