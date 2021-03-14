% Read in two desired images
img_A = imread('./data/EiffelA.png');
img_B = imread('./data/EiffelB.png');

% Read in extracted and matched features
load('./data/Eiffel_vpts.mat');
data = validation.pts;
% Read in true homography matrix for validation against ground truth
H_true = validation.model;

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