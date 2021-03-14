% Lambda is the tuning parameter multiplied with our L1 penalty term
LAMBDA = 10;

% Read in two desired images
img_A = imread('./data/CapitalRegionA.jpg');
img_B = imread('./data/CapitalRegionB.jpg');

% Read in extracted and matched features
load('./data/CapitalRegion_vpts.mat');
data = validation.pts;
% Read in true homography matrix for validation against ground truth
H_true = validation.model;

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
    minimize 1/num_points*pow_pos(norm(A_est - A_true),2) + LAMBDA*norm(B_true,1)
cvx_end

train_error = cvx_optval;
test_error = norm(H_est - H_true)^2 / 9;

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
        A_point = H_est*B_point;
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