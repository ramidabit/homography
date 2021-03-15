% NUM_ITER is the number of iterations RANSAC will run
NUM_ITER = 20;
% MAX_ERR is the maximum distance allowed for a point to be an inlier
MAX_ERR = 3;

% Read in two desired images
img_A = imread('./data/EiffelA.png');
img_B = imread('./data/EiffelB.png');

% Read in extracted and matched features
load('./data/Eiffel_vpts.mat');
data = validation.pts;
% Read in true homography matrix for validation against ground truth
H_true = validation.model;

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
        A_point = best_H*B_point;
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