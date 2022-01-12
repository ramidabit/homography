# Homography Estimation for Image Alignment

Hello and welcome to my homography project!

This project explores one particularly puzzling problem in computer vision: how to align two images for an increased field of view. Given two photos taken from the same viewpoint, one should be able to map points from one image to the other by multiplication with what is known as the homography matrix. Estimation of the matrix, however, is an optimization problem often solved using least squares regression. By detecting, extracting, and matching a few features which are present in both images, we can attempt to project all the points from one image onto the other. Meanwhile, we also wish to minimize the distance between their projected location and where they actually reside. If the mapping is approximated correctly, applying the homography to one of the images should cause it to rotate and translate such that it perfectly aligns with the other image, and overlaying the two images should produce a beautiful panorama. This will be achieved here using MATLAB and the CVX library for convex optimization.


Please visit the project website located below:

https://ramidabit.github.io/homography/
