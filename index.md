## **Table of Contents**
1. [Introduction](#1-introduction)
   - A. [Problem Overview](#a-problem-overview)
   - B. [The Dataset](#b-the-dataset)
   - C. [Expectations](#c-expectations)

2. [Least Squares Regression](#2-least-squares-regression)
   - A. [Mathematical Model](#a-mathematical-model)
   - B. [Solution](#b-solution)
   - C. [Results and Discussion](#c-results-and-discussion)

3. [L1 Regression](#3-l1-regression)
   - A. [Mathematical Model](#a-mathematical-model-1)
   - B. [Solution](#b-solution-1)
   - C. [Results and Discussion](#c-results-and-discussion-1)

4. [Regularized Regression](#4-regularized-regression)
   - A. [Mathematical Model](#a-mathematical-model-2)
   - B. [Solution](#b-solution-2)
   - C. [Results and Discussion](#c-results-and-discussion-2)

5. [Random Sample Consensus](#5-random-sample-consensus)
   - A. [Mathematical Model](#a-mathematical-model-3)
   - B. [Solution](#b-solution-3)
   - C. [Results and Discussion](#c-results-and-discussion-3)

6. [k-fold Cross-Validation](#6-k-fold-cross-validation)
   - A. [Mathematical Model](#a-mathematical-model-4)
   - B. [Solution](#b-solution-4)
   - C. [Results and Discussion](#c-results-and-discussion-4)

7. [Conclusion](#7-conclusion)


## **1. Introduction**
One puzzling problem in computer vision is how to align two images such that they may be overlaid for an increased field of view. Given two photos taken from the same viewpoint, one should be able to map points from one image to the other by multiplication with what is known as the homography matrix. Estimation of the matrix, however, is an optimization problem often solved using least squares regression. By detecting, extracting, and matching a few features which are present in both images, we can attempt to project all the points from one image onto the other. Meanwhile, we also wish to minimize the distance between their projected location and where they actually reside. If the mapping is approximated correctly, applying the homography to one of the images should cause it to rotate and translate such that it perfectly aligns with the other image, and overlaying the two images should produce a beautiful panorama. We will attempt to do so here using MATLAB and the CVX library for convex optimization.

Given that the regression task is inherently susceptible to outliers, we must also explore other approaches for robust homography estimation. Instead of minimizing the 2-norm as in least squares, we may choose to minimize the 1-norm, or perform L1 regression. Outlying matches may also be penalized using regularized regression methods such as Lasso and Ridge, also respectively known as L1 and L2 regularization. Alternatively, we may use an iterative approach such as the random sample consensus (RANSAC) algorithm to reject outliers, or even k-fold cross-validation for a more completely-sampled exploration of our data. The goal of this project is to estimate homographies using the using these various methods, and to compare and contrast their performance by attempting to stitch photos together using the estimated mapping.

### A. Problem Overview
**What is a Homography?**

The solution to the image alignment problem involves a series of rotations and translations in what is generally known as a projective transformation; however, a projective transformation in two dimensions can simply be called a homography. In a homography, each image point has a third dimension added to it, moving from the realm of image coordinates to homogeneous coordinates. Hence, multiplication by the homography matrix provides a mapping from ℝ<sup>3</sup> to ℝ<sup>3</sup>.

![Image](https://i.imgur.com/dfwXYqP.png)

This involves a third element, _w_, which is added to allow the affine transformation: a subclass of projective transformations which includes linear transformations as well as nonlinear, affine operations. The key to our homography matrix is that it is essentially an affine transformation, where both lines and parallelism are preserved from one image plane to the other. Generally speaking, this is not true for projective transformations, which preserve lines but not necessarily parallelism.

![Image](https://i.imgur.com/0uukr5N.png)

Following our mapping of points from one image to the next, we must finally convert out of homogeneous coordinates by normalizing by the third coordinate, _w'_. This gives a resulting point [x'/w', y'/w', 1]<sup>T</sup>, where we can simply drop the ‘1’ and return to an image point in ℝ<sup>2</sup>.

### B. The Dataset
The dataset for this project is provided by the Czech Technical University in Prague, and includes several image pairs whose features have already been extracted and matched. Along with each image pair, the correct homography matrix is given so may verify our work. As such, each implementation will be followed by a testing error, or mean squared distance between the estimated and true homography matrices. While it is important to report these for a quantitative measurement of our performance, it is also important to note that these values should hold little weight. From the "README" file of the dataset, the point correspondences were manually selected by the creator who zoomed in to select features and their corresponding matches. Hence, the accuracy of the author is a factor in determining the "correct" mapping of the points. In many of the following cases, we see near-zero testing errors as well as misalignment, giving further cause as to why we should visually verify our images rather than relying on metrics alone.

The “homogr” dataset for homographies is available at [CMP :: Data :: Two-view geometry](http://cmp.felk.cvut.cz/data/geometry2view/index.xhtml).

### C. Expectations
Throughout the following approaches, we will apply our homography to three images titled "Boston," "Capital Region," and "Eiffel." Prior to estimating the homography, we must import the data using one of the following code blocks.

**Boston:**
```markdown
% Read in two desired images
img_A = imread('./data/BostonA.jpg');
img_B = imread('./data/BostonB.jpg');

% Read in extracted and matched features
load('./data/Boston_vpts.mat');
data = validation.pts;
% Read in true homography matrix for validation against ground truth
H_true = validation.model;
```

**Capital Region:**
```markdown
% Read in two desired images
img_A = imread('./data/CapitalRegionA.jpg');
img_B = imread('./data/CapitalRegionB.jpg');

% Read in extracted and matched features
load('./data/CapitalRegion_vpts.mat');
data = validation.pts;
% Read in true homography matrix for validation against ground truth
H_true = validation.model;
```

**Eiffel:**
```markdown
% Read in two desired images
img_A = imread('./data/EiffelA.png');
img_B = imread('./data/EiffelB.png');

% Read in extracted and matched features
load('./data/Eiffel_vpts.mat');
data = validation.pts;
% Read in true homography matrix for validation against ground truth
H_true = validation.model;
```

Using either the given or estimated homography matrix, we can transform each image and present their overlay using the following code. In this first example, "H_true" is used in place of "H_est" to obtain the expected ground truths.

```markdown
% Want to extend the dimensions of image A by the diagonal
%  of image B, which is the max amount they may overlap
dist_extend = ceil(sqrt(size(img_B,1)^2 + size(img_B,1)^2));
% Extend image A from the left and from above
A_extended = imtranslate(img_A,[dist_extend,dist_extend],'FillValues',127,'OutputView','full');
% Extend image A from the right and from below
A_extended = imtranslate(A_extended,[-dist_extend,-dist_extend],'FillValues',127,'OutputView','full');
% Add 3 more dimensions to image A to hold the color channels for image B
A_extended = cat(3,A_extended,127*ones(size(A_extended)));

% Need to find valid area for display
smallest_x = intmax;
greatest_x = intmin;
smallest_y = intmax;
greatest_y = intmin;

for x = 1:size(img_B,2)
    for y = 1:size(img_B,1)
        % Convert point in image B to homogeneous coordinates
        B_point = [x; y; 1];
        % Use homography matrix to map point in image B to image A
        A_point = H_true*B_point;
        % Normalize point in image A by third coordinate
        A_point = A_point / A_point(3);
        % Drop third coordinate to convert from homogeneous to image coords
        A_point(3) = [];
        % Eliminate floating point values by rounding down
        A_point = floor(A_point);
        
        % Find bounds of the transformed image
        if (A_point(1) + dist_extend) < smallest_x
            smallest_x = A_point(1) + dist_extend;
        end
        if (A_point(1) + dist_extend) > greatest_x
            greatest_x = A_point(1) + dist_extend;
        end
        if (A_point(2) + dist_extend) < smallest_y
            smallest_y = A_point(2) + dist_extend;
        end
        if (A_point(2) + dist_extend) > greatest_y
            greatest_y = A_point(2) + dist_extend;
        end
        
        % Place the projection of image B in the last three channels of image A
        A_extended(A_point(2)+dist_extend, A_point(1)+dist_extend, 4:6) = img_B(y,x,:);
    end
end

% Original image A is in dist_extend+1:dist_extend+1+size(img_A,1),
%  dist_extend+1:dist_extend+1+size(img_A,2)
% Transformed image B is in smallest_y:greatest_y, smallest_x:greatest_x

beg_row = min(dist_extend+1,smallest_y);
end_row = max(dist_extend+1+size(img_A,1),greatest_y);
beg_col = min(dist_extend+1,smallest_x);
end_col = max(dist_extend+1+size(img_A,2),greatest_x);

% Select only the valid area of projected image
valid_A = A_extended(beg_row:end_row,beg_col:end_col,1:3);
valid_B = A_extended(beg_row:end_row,beg_col:end_col,4:6);

% Overlay the two images for display
figure, imshowpair(valid_A,valid_B,'blend');
```

The ground truth images, which we hope to validate our estimates against, are given below.


**Boston:**
![Image](https://i.imgur.com/efXB6tk.png)

**Capital Region:**
![Image](https://i.imgur.com/FwNg0Uq.png)

**Eiffel:**
![Image](https://i.imgur.com/4Eb6WSr.png)


Note that the results are imperfect even when using the given homography matrix.


## **2. Least Squares Regression**
### A. Mathematical Model
As we have already seen above, multiplication by a 3-by-3 homography matrix provides a mapping from points in one image plane to another, and this mapping is in fact affine. In an affine transformation, not all nine elements of the transformation matrix are unknown. Namely, the bottom row is already known to be [0 0 1], where the bottom right element is a scaling factor between the two images. We are hence left with six degrees of freedom, requiring only three feature matches to estimate the entire homography. While I was able to model this in CVX as an unconstrained optimizaiton problem with no issues, it is more correct to assign these constraints on the elements of the bottom row.
![Image](https://i.imgur.com/dOE02vX.png)

The crux of the problem is minimizing the residuals between the estimated point mapping and the actual destination of the point's coordinates. Given that the input data is already in homogeneous coordinates—with a third coordinate of 1—this may be why CVX is able to correctly identify the bottom row in the absence of these constraints. With all nine elements of the homography matrix as our decision variables, we may formulate the optimization problem as follows.
![Image](https://i.imgur.com/E6GzbeW.png)

### B. Solution
We can solve this optimization in CVX using the code block below, which is also wrapped in the data import and image output blocks introduced previously.
```markdown
% First three rows contain points in image A
A_true = data(1:3,:);
% Last three rows contain corresponding points in image B
B_true = data(4:6,:);
num_points = size(B_true,2);

% Estimate homography matrix, H, using least squares regression
cvx_begin
    variable H_est(3,3)
    expression A_est(size(A_true))
    
    for i = 1:num_points
        A_est(:,i) = H_est*B_true(:,i);
    end
    minimize pow_pos(norm(A_est - A_true),2) / num_points
    
    subject to
    H_est(3,1) == 0;
    H_est(3,2) == 0;
    H_est(3,3) == 1;
cvx_end

train_error = cvx_optval;
test_error = norm(H_est - H_true)^2 / 9;
```

### C. Results and Discussion
Below are the results of the homography on our three test images using least squares regression.


**Boston:**
![Image](https://i.imgur.com/ekdQ5JO.png)
- Training Error: 95.2043
- Test Error: 34.7323

Compared to the ground truth homography, our estimate contains immediately noticable misalignment throughout the image. Zooming in on the borders of the images, we see trouble areas caused by the inaccuracy of our estimated homography matrix, namely in the waterfront on the left and the boat on the right. Looking at the top of the buildings also showcases the inaccuracy, which we hope to reduce using the methods below.

**Capital Region:**
![Image](https://i.imgur.com/ezvSvLA.png)
- Training Error: 77.3935
- Test Error: 6.6612e+04

In the case of this overhead image, our model still does a great job of continuing the roads from one photo to the next, although it becomes obvious that the estimate is misaligned when we look at some of the city blocks. For example, the snowy block in the center of the image appears very shaky in the estimate, whereas this effect is less severe when using the ground truth homography. It is also interesting to notie that only one photo contains the wing of the plane, which leads to a "cut off" wing in the overlay.

**Eiffel:**
![Image](https://i.imgur.com/Po8iKcT.png)
- Training Error: 22.2534
- Test Error: 8.5642e+03

At first glance, we might think that this estimate actually looks better than the ground truth, which still has quite a bit of misalignment and suffers greatly from artifacts created when warping the image. However, taking a closer look at the center of the tower, we see that the estimate is less-closely aligned compared to the ground truth.


## **3. L1 Regression**
### A. Mathematical Model
Rather than minimizing the 2-norm or Euclidian distance as we did in the previous example, we can instead minimize the 1-norm or absolute deviations. Called L1 regression, this approach is known to be more robust to outliers because it generally leads to more sparse solutions, weighing less important features accordingly. It can be modeled as the following optimization problem.
![Image](https://i.imgur.com/IFKaFRR.png)

### B. Solution
Similar to the previous approach, we can solve this optimization problem using CVX as implemented in the code below.
```markdown
% First three rows contain points in image A
A_true = data(1:3,:);
% Last three rows contain corresponding points in image B
B_true = data(4:6,:);
num_points = size(B_true,2);

% Estimate homography matrix, H, using L1 regression
cvx_begin
    variable H_est(3,3)
    expression A_est(size(A_true))
    
    for i = 1:num_points
        A_est(:,i) = H_est*B_true(:,i);
    end
    minimize pow_pos(norm(A_est - A_true,1),2) / num_points
    
    subject to
    H_est(3,1) == 0;
    H_est(3,2) == 0;
    H_est(3,3) == 1;
cvx_end

train_error = cvx_optval;
test_error = norm(H_est - H_true)^2 / 9;
```

### C. Results and Discussion
Below are the results of the homography in our three test cases using L1 regression.


**Boston:**
![Image](https://i.imgur.com/qOW7U7h.png)
- Training Error: 10.6060
- Test Error: 86.0373

**Capital Region:**
![Image](https://i.imgur.com/EZxiDPf.png)
- Training Error: 31.0400
- Test Error: 6.8106e+04

**Eiffel:**
![Image](https://i.imgur.com/mAWwGRa.png)
- Training Error: 8.8067
- Test Error: 7.8049e+03

In attemtping to improve our estimate with L1 regression, we actually find that the the Boston case looks marginally better in the least squares approach, where there is less misalignment toward the tops of the buildings. This is counterintuitive as we would expect that minimizing the sum of squared differences would be more susceptible to outliers, although the difference is not too severe. As for the Capital Region, the L2 case looks slightly better when paying attention to the connections of the roads along borders of each image. The Eiffel case confirms that L1 regression is worse than the L2 approach when looking at the arch at the underside of the tower, where the curve sticks out slightly where it was previously hidden in its entirety. This is surprising but makes sense as the L1 regression approach is less complex than least squares, giving a more general fit which may lead to misalignment.


## **4. Regularized Regression**
### A. Mathematical Model
Outliers may sometimes cause our model to select values for the elements of _H_ which are too high, and we can penalize against these effects by adding a penalty term to our least squares model. The penatly term comes in the form of the decision variable added to our objective function and with a norm taken of it, and is also multiplied by a tunable parameter λ. In this section, we will explore two forms of regularization called Lasso and Ridge regularization, or L1 and L2 regularization, respectively.

**Lasso (L1) Regularization:**
![Image](https://i.imgur.com/a53N7G6.png)
**Ridge (L2) Regularization:**
![Image](https://i.imgur.com/K4kUvVF.png)

### B. Solution
Lasso and Ridge regularization are implemented in CVX using the code blocks below.
**Lasso:**
```markdown
% Lambda is the tuning parameter multiplied with our L1 penalty term
LAMBDA = 0.5;

% First three rows contain points in image A
A_true = data(1:3,:);
% Last three rows contain corresponding points in image B
B_true = data(4:6,:);
num_points = size(B_true,2);

% Estimate homography matrix, H, using Lasso regression
cvx_begin
    variable H_est(3,3)
    expression A_est(size(A_true))
    
    for i = 1:num_points
        A_est(:,i) = H_est*B_true(:,i);
    end
    minimize 1/num_points*pow_pos(norm(A_est - A_true),2) + LAMBDA*norm(H_est,1)
    
    subject to
    H_est(3,1) == 0;
    H_est(3,2) == 0;
    H_est(3,3) == 1;
cvx_end

train_error = cvx_optval;
test_error = norm(H_est - H_true)^2 / 9;
```
**Ridge:**
```markdown
% Lambda is the tuning parameter multiplied with our L2 penalty term
LAMBDA = 0.001;

% First three rows contain points in image A
A_true = data(1:3,:);
% Last three rows contain corresponding points in image B
B_true = data(4:6,:);
num_points = size(B_true,2);

% Estimate homography matrix, H, using Ridge regression
cvx_begin
    variable H_est(3,3)
    expression A_est(size(A_true))
    
    for i = 1:num_points
        A_est(:,i) = H_est*B_true(:,i);
    end
    minimize 1/num_points*pow_pos(norm(A_est - A_true),2) + LAMBDA*pow_pos(norm(H_est),2)
    
    subject to
    H_est(3,1) == 0;
    H_est(3,2) == 0;
    H_est(3,3) == 1;
cvx_end

train_error = cvx_optval;
test_error = norm(H_est - H_true)^2 / 9;
```

### C. Results and Discussion
Below are the results of projection for the three cases using Lasso and Ridge Regularization.


**Boston:**

**Lasso, λ = 0.5:**
![Image](https://i.imgur.com/B8c4kvy.png)
- Training Error: 441.7389
- Test Error: 166.3076

**Ridge, λ = 0.001:**
![Image](https://i.imgur.com/5C06L1W.png)
- Training Error: 578.4854
- Test Error: 88.3699

**Capital Region:**

**Lasso, λ = 0.5:**
![Image](https://i.imgur.com/Sj0xpPy.png)
- Training Error: 759.4370
- Test Error: 7.1245e+04

**Ridge, λ = 0.001:**
![Image](https://i.imgur.com/l5dObPN.png)
- Training Error: 1.3704e+03
- Test Error: 7.3165e+04

**Eiffel:**

**Lasso, λ = 0.5:**
![Image](https://i.imgur.com/OiLaoSD.png)
- Training Error: 607.9034
- Test Error: 1.0234e+04

**Ridge, λ = 0.001:**
![Image](https://i.imgur.com/akjdjGm.png)
- Training Error: 658.2836
- Test Error: 1.4798e+04

The tuning of the parameter λ is critical to both regularization methods, with higher λ values causing incredibily inaccurate results. This was especially noticable in the case of Ridge regularization, where λ had to be reduced to 0.001. However, we do ultimately notice better results than the previous two approaches. Looking at the bottom left of the Boston image, we see more overlap in the white portions of the sails in both the Lasso and the Ridge case. With Lasso regularization, the roads look quite good in the Capital Region example, although the Ridge case shows more misalignment. Using the same approach on the Eiffel image, we begin to see the increased warping in the image on the left, which is initially a good sign because this is also present in the ground truth. However, the underside of the tower is clearly misaligned, and more so than the least squares and L1 regression approaches.

Between Lasso and Ridge, it appears that Lasso is capable of giving consistently good results, with all three images showing improvement over the previous implementations. Although the least squares Eiffel output looks the most correct, Lasso appears closest to the ground truth, and the extremely low test error is confirmation of that. Meanwhile, Ridge regularization is more sensitive to changes in λ, meaning we can achieve more precise alignment if we are willing to fine-tune the parameter by hand, although its performance may vary on a case-to-case basis seeing as it performed worse in the Capital Region case.


## **5. Random Sample Consensus**
### A. Mathematical Model
Unlike our previous examples where the homography matrix is found using an optimzation model, random sample consensus (RANSAC) works iteratively to find an appropriate homography. In each iteration, three random point-pairs are randomly selected from our dataset. With these three datapoints, RANSAC hypothesizes a homography matrix which might describe the mapping from one image to the other. Using this hypothesis, the error value for the remaining datapoints is calculated, and the number inliers is counted. At the end of the interations, the model with the highest number of inliers is chosen as the correct mapping.

In the implementation below, we feed RANSAC with two parameters: a predefined number of iterations and the maximum amount error which may be tolerated for a datapoint to count as an inlier. Typically, the number of iterations needed depends on multiple factors, such as the desired probability of the final result being correct and the ratio of inliers to outliers in the dataset; however, we can also predefine it for simplicity. RANSAC works best when the outlier ratio is less than 50% of the available datapoints.

### B. Solution
Below is the implementation of random sample consensus, which uses CVX to perform least squares at each iteration. The datapoints which are not randomly selected to train on are instead responsible for "voting" on the correctness of the chosen model by supplying a test error.

Note that 30 and 5 are shown for the number of iterations and maximum error below, but these parameters may be adjusted. These values correspond to the first case of "Boston," and are different in the cases "Capital Region" and "Eiffel."
```markdown
% NUM_ITER is the number of iterations RANSAC will run
NUM_ITER = 30;
% MAX_ERR is the maximum distance allowed for a point to be an inlier
MAX_ERR = 5;

% Only need three points to compute a homography
num_points = 3;

best_H = ones(3,3);
best_num_inliers = 0;
best_train_error = intmax;

% Random Sample Consensus (RANSAC)
for iter = 1:NUM_ITER    
    % Randomly choose three potential inliers from the extracted keypoints
    chosen_col = randperm(size(data,2),num_points);
    A_chosen = data(1:3,chosen_col);
    B_chosen = data(4:6,chosen_col);
    
    % Save the remaining points for validation (counting number of inliers)
    data_remaining = data;
    data_remaining(:,chosen_col) = [];
    A_remaining = data_remaining(1:3,:);
    B_remaining = data_remaining(4:6,:);
    
    % Estimate homography matrix using least squares with the chosen points
    cvx_begin
        variable H_est(3,3)
        expression A_est(size(A_chosen))

        for i = 1:num_points
            A_est(:,i) = H_est*B_chosen(:,i);
        end
        minimize 1/num_points * pow_pos(norm(A_est - A_chosen),2)
        
        subject to
        H_est(3,1) == 0;
        H_est(3,2) == 0;
        H_est(3,3) == 1;
    cvx_end
    train_error = cvx_optval;
    
    % Validate the model on all other points by finding the distance
    %  between estimate and ground truth
    num_remaining = size(data_remaining,2);
    num_inliers = 0;
    for i = 1:num_remaining
        A_est(:,i) = H_est*B_remaining(:,i);
        distance = 1/num_remaining * norm(A_est(:,i) - A_remaining(:,i))^2;
        if distance < MAX_ERR
            num_inliers = num_inliers + 1;
        end
    end
    
    % Save the best homography matrix according to number of inliers
    if num_inliers > best_num_inliers
        best_H = H_est;
        best_num_inliers = num_inliers;
        best_train_error = train_error;
    end
end

num_inliers = best_num_inliers;
train_error = best_train_error;
test_error = 1/9*norm(H_est - H_true)^2;
```

### C. Results and Discussion
Below are the results produced by RANSAC for each of our test image pairs.

**Boston:**
![Image](https://i.imgur.com/8lE5ost.png)
- Number of Iterations: 30
- Maximum Error: 5
- Number of Inliers: 2
- Training Error: 4.4985e-09
- Test Error: 106.9545

Based on the results above, RANSAC appears to perform much better than any of our previous implementations, which is not surpising given that is is the current gold standard for image alignment. However, this is not to be celebrated too early as alignment in some areas appears to cause misalignment in others. Looking closer at the Boston example, the tops of the buildings are near-perfectly aligned, although we see less alignment in the sails found at the bottom-left. We also notice more shakiness in the projection of the tent, which we had stabilized using both Lasso and Ridge.

**Capital Region:**
![Image](https://i.imgur.com/oypzwln.png)
- Number of Iterations: 60
- Maximum Error: 4
- Number of Inliers: 1
- Training Error: 7.7442e-09
- Test Error: 6.3250e+04

The Capital Region case shows the most promise with respect to RANSAC, with roads that are perfectly aligned and a center snowy block that looks almost as good as the ground truth. The road in the upper-left triangle connects seamlessly across the two images, and both the training and test error are extremely low in this example. It is slightly concerning that such a good result was chosen with a single inlier, but is not suprising given that we have only eight datapoints, five of which are left for testing.

**Eiffel:**
![Image](https://i.imgur.com/ItzYZcA.png)
- Number of Iterations: 20
- Maximum Error: 3
- Number of Inliers: 2
- Training Error: 7.7442e-09
- Test Error: 1.3132e+04

The Eiffel example also shows a lot of promise, with a result that looks even better than the ground truth when inspecting the underside of the tower. However, looking at the left side of the tower's leg, we notice that it has misaligned in the opposite direction. Although there are less artifacts in the left image compared to the ground truth, this does not seem to be a worthwhile tradeoff.

We now see that a higher number of inliers doesn't necessarily guarantee a better looking output. The Capital Region case has a higher test error than Eiffel, but looks better to the human eye. Nevertheless, it appears that choosing a few samples and validating against the rest of the data is a very good approach to producing good-looking, well-aligned images.

It is important to note that smaller training errors were observed as least squares was performed at each iteration, although the model with the smallest training error did not yield the best result. This is an inherent flaw of RANSAC, which uses the number of inliers as its metric for the best mapping rather than the training error. Also, the fact that we are taking random samples leads to inconsistent results unless you run the program for an extremely high number of iterations. Let's see if we can solve both of these issues by performing k-fold cross-validation.


## **6. k-fold Cross-Validation**
### A. Mathematical Model
In performing random sample consensus, I was inspired to implement k-fold cross-validation due to the similarity of the approaches. In k-fold cross-validation, the dataset is partitioned into k subsets where each subset is used exactly once as the training set, with the remainder of the datapoints used for training. Rather than choosing random samples, k-fold cross validation offers a more complete coverage of the dataset since every point-pair is used to train with at least once. Like RANSAC, we maintain the least squares approach which runs in every iteration.

The question becomes what is our choice of k? With few points needed and few points to choose from, we find that there are "number of points" choose "3" possible combinations of point-pairs that could give a homography. As such, we train on all of these combinations one time, and use the error given by the rest of the points as our metric for choosing the best matrix, _H_. Hypothetically, this purely quantitative measurement should give better results than the notion of "inliers" used by RANSAC, and our results are expected to reflect that. We should also expect the best result each and every time since we are no longer relying on random samples.

### B. Solution
An implementation of k-fold cross-validation using CVX at each iteration is given below.
```markdown
% Only three points are needed to compute a homography. Hence, k depends
%  on the number of 3-point combinations possible with the given data
num_points = size(data,2);
k = nchoosek(num_points,3);

% Partition data into k sets, each containing 3 point-pairs for training,
%  as well as the remainder of the data for validation
subsets = zeros(size(data,1),size(data,2),k);
indices = nchoosek(1:num_points,3);
for set = 1:k
    % First three columns of this subset contain the chosen point-pairs
    subsets(:,1,set) = data(:,indices(set,1));
    subsets(:,2,set) = data(:,indices(set,2));
    subsets(:,3,set) = data(:,indices(set,3));
    % Delete the columns that have already been chosen
    data_remaining = data;
    data_remaining(:,indices(set,1)) = [];
    data_remaining(:,indices(set,2)-1) = [];
    data_remaining(:,indices(set,3)-2) = [];
    % Remaining columns of this subset contain the remaining point-pairs
    subsets(:,4:num_points,set) = data_remaining;
end

best_H = ones(3,3);
best_train_error = intmax;
best_test_error = intmax;

% k-fold cross validation
for iter = 1:k
    % Estimate homography matrix using least squares with the current subset
    A_chosen = subsets(1:3,1:3,iter);
    B_chosen = subsets(4:6,1:3,iter);
    cvx_begin
        variable H_est(3,3)
        expression A_est(size(A_chosen))

        for i = 1:3
            A_est(:,i) = H_est*B_chosen(:,i);
        end
        minimize 1/3 * pow_pos(norm(A_est - A_chosen),2)
        
        subject to
        H_est(3,1) == 0;
        H_est(3,2) == 0;
        H_est(3,3) == 1;
    cvx_end
    train_error = cvx_optval;
    
    % Validate the model using all remaining points in the current subset
    A_remaining = subsets(1:3,4:num_points,iter);
    B_remaining = subsets(4:6,4:num_points,iter);
    num_remaining = num_points - 3;
    distance = zeros(1,num_remaining);
    for i = 1:num_remaining
        A_est(:,i) = H_est*B_remaining(:,i);
        distance(i) = norm(A_est(:,i) - A_remaining(:,i));
    end
    test_error = 1/num_remaining * sumsqr(distance);
    
    % Save the best homography matrix based on the test error for this set
    if test_error < best_test_error
        best_H = H_est;
        best_train_error = train_error;
        best_test_error = test_error;
    end
end

train_error = best_train_error;
test_error = best_test_error;
H_error = 1/9*norm(H_est - H_true)^2;
```

### C. Results and Discussion
Following k-fold cross-validation, the following image overlays were produced.

**Boston:**
![Image](https://i.imgur.com/LhtAmug.png)
- Training Error (Best case): 7.7442e-09
- Test Error (Best case): 97.3498
- Test Error (Against True H): 51.7504

**Capital Region:**
![Image](https://i.imgur.com/j7BkREG.png)
- Training Error (Best case): 7.7442e-09
- Test Error (Best case): 232.4625
- Test Error (Against True H): 5.7804e+04

**Eiffel:**
![Image](https://i.imgur.com/iM6kvuj.png)
- Training Error (Best case): 7.7442e-09
- Test Error (Best case): 69.2201
- Test Error (Against True H): 5.3474e+03

I was surprised to find that k-fold cross validation performed worse than RANSAC, considering that RANSAC randomly chooses three datapoints to potentially describe our affine mapping while k-fold cross validation tries them all. It turns out that counting the number of inliers is a viable method after all, whereas minimizing the testing error is less effective. This is likely due to the fact that if our dataset contains outliers, and we are minimizing the distance to those outliers, then we will end up choosing the mapping which most closely resembles the outlying matches.

The misalignment in the Boston case appears to be inconsistent here, with the tallest skyscraper and other buildings perfectly aligned, but misaligned buildings in between them. We also notice shakiness in the waterfront, and the leftmost sails no longer overlap. The Capital Region shot that previously captured hearts is shaky once again, and we find that even the road underneath the airplane wing is now diverging. On the other hand, the Eiffel example still looks quite good, and may even be comparable to the ground truth if not for the clearly misaligned undercarriage of the tower.


## **7. Conclusion**
While it is the one examined approach that does not stem directly from an optimization problem, the iterative random sample consensus algorithm is the clear winner among the various implementations above. It seems as though it is the industry standard for a reason, and given enough iterations, will produce the best result which contains the highest number of inliers. While it produced very good results across all three examples, the choice of approach still seems to depend on which features one would like to align. For example, L1 regression best-algined all of the Boston buildings, whereas RANSAC nailed the alignment of the main skyscraper. Lasso and Ridge regression did a great job aligning the outer part of the leg in Eiffel, and also produced some similar warping to that of the ground truth.

In a future application, it would be interesting to see how well RANSAC performs when combinining it with Lasso or Ridge regularization at each iteration, namely Ridge due to the sensitivity of the result with the λ paramater. Both RANSAC and Ridge regularization contain parameters which may be very finely tuned if the user is willing to put in the time and effort, making them well-suited options for achieving that perfect alignment for a panorama. Given more time, I would have liked to attempt it with the two images taken below, although it would also require extracting and matching my own features. Rather than doing it by hand, I would use MATLAB's built in libraries for feature descriptors such as SURF, BRISK, or ORB. I would then match them using FLANN or the Hamming distance between keypoints in the two images. All of these options are available in the computer vision toolbox, and would make a great follow-up project if I were to produce more of my own data.

![Image](https://i.imgur.com/xSDuva3.jpg)
