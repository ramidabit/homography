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

**Capital Region:**
![Image](https://i.imgur.com/ezvSvLA.png)
- Training Error: 77.3935
- Test Error: 6.6612e+04

**Eiffel:**
![Image](https://i.imgur.com/Po8iKcT.png)
- Training Error: 22.2534
- Test Error: 8.5642e+03


## **3. L1 Regression**
### A. Mathematical Model
Rather than minimizing the 2-norm or Euclidian distance as we did in the previous example, we can instead minimize the 1-norm or absolute deviations. Called L1 regression, this approach is known to be more robust to outliers because it generally leads to more sparse solutions, weighing less important features accordingly. It can be modeled as the following optimization problem.
![Image](https://i.imgur.com/IFKaFRR.png)

### B. Solution
Similar to the previous approach, we solve this optimization problem using CVX as implemented in the code below.
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


## **4. Regularized Regression**
### A. Mathematical Model
Outliers may sometimes cause our model to select values for the elements of _H_ which are too high or otherwise erroneous, and we can penalize against these effects by adding a penalty term to our least squares model. The penatly term comes in the form of an additional decision variable, which we will call _h_, has a norm taken of it, and is also multiplied by a tunable parameter λ. In this section, we will explore two forms of regularization called Lasso and Ridge regularization, or L1 and L2 regularization, respectively.
![Image]()
![Image]()

### B. Solution
### C. Results and Discussion


## **5. Random Sample Consensus**
### A. Mathematical Model
### B. Solution
### C. Results and Discussion


## **6. k-fold Cross-Validation**
### A. Mathematical Model
### B. Solution
### C. Results and Discussion


## **7. Conclusion**
