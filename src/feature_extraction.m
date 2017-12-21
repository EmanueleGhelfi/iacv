%% clear all
clc
clear all
debug = false; % do not diplay a thousand of images
% Use Canny with contrast adjustment
%% Open the image
im_rgb = imread('Input image.jpeg');

if debug
    figure, imshow(im_rgb);
end

%% First convert the image in greyscale
im_grey = double(rgb2gray(im_rgb))./255;
if debug
    figure, imshow(im_grey);
end
%% adjust contrast
im_grey_adj = imadjust(im_grey);
if debug
    figure
    imshow(im_grey_adj)
end

%% Edge detection with canny
% sobel, canny, prewitt, roberts
% BW2 = edge(im_grey,'canny');
[BW2, th] = edge(im_grey,'canny');
if debug
    figure;
    imshow(BW2)
    title('Canny Filter');
end
th

%% Change Canny Threshold
% try changing threshold *1.5 is good
th2 = th.*[0.2, 2];
BW2 = edge(im_grey,'canny', th2);
if debug
    figure, imshow(BW2);
end

%% with roberts
[BW2, th] = edge(im_grey,'roberts');
figure;
imshow(BW2)
title('Roberts Filter');
%% Change ROberts Threshold
% try changing threshold
th2 = th*0.7
BW2 = edge(im_grey,'roberts', th2);
figure, imshow(BW2);
%% Sobel
[BW2, th] = edge(im_grey,'Sobel', 0.035);
figure;
imshow(BW2)
title('Sobel Filter');
th

%% Log
[BW2, th] = edge(im_grey,'log', 0.0025);
figure;
imshow(BW2)
title('Log Filter');
th
%% Line Detection
% Hough parameters
minLineLength_vert = 170;
fillGap = 20;
numPeaks = 100;
NHoodSize = [111 81];
vertical_angles = -90:0.5:89.8;

% find lines vertical
[H,theta,rho] = hough(BW2,'RhoResolution', 1, 'Theta', vertical_angles);
% find peaks in hough transform
P = houghpeaks(H,numPeaks,'threshold',ceil(0.05*max(H(:))), 'NHoodSize', NHoodSize);
%P = houghpeaks(H,numPeaks,'threshold',ceil(0.2*max(H(:))));

% find lines using houghlines
lines = houghlines(BW2,theta,rho,P,'FillGap',fillGap,'MinLength',minLineLength_vert);
%lines = houghlines(BW2,theta,rho,P);

% plot lines
figure, imshow(im_rgb), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',0.5,'Color','green');

   % Plot beginnings and ends of lines
   %plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   %plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   text(xy(1,1),xy(1,2), num2str(k), 'Color', 'red')
   %text(xy(1,1),xy(1,2),strcat(num2str(xy(1,1)),",",num2str(xy(1,2))), 'Color', 'red')
end

%% Stratification approach:
% - compute affine reconstruction
% l_inf -> l_inf
% H_r_aff = [1  0  0
  %          0  1  0
  %          l1 l2 l3] where the last line is l_inf'

% Extract parallel lines
line_indices = [1, 2, 1, 18, 18, 2, 55, 31, 55, 15, 31, 15, 4, 24, 39, 4, 39, 24, 13, 24, 50, 19, 19, 16, 68, 33, 3, 24, 3, 39];

% in this we save parallel lines 
parallelLines = zeros(3, size(line_indices,2));

for ii = 1:size(line_indices,2)
    parallelLines(:, ii) =  getLine(lines(line_indices(ii)).point1, lines(line_indices(ii)).point2);
end

%% extract the line at inf prim

l_inf_prime = getL_inf_prime(parallelLines)

%% compute H_r_aff (as before)

H_r_aff = [1 0 0; 0 1 0; l_inf_prime(1) l_inf_prime(2) l_inf_prime(3)];

% Transform the image
tform_r_aff = projective2d(H_r_aff.');
outputImage_aff = imwarp(im_rgb, tform_r_aff);
figure();
imshow(outputImage_aff);
hold on
%% Metric rectification
% perpendicular line indices
perp_line_indices = [1, 31, 31, 2, 4, 33, 24, 68, 67, 16, 31, 8, 2, 55, 68, 3, 39, 33, 62, 39, 67, 19, 55, 1, 55, 18, 31, 18, 24, 33, 67, 50];

% in this we save perpendicular lines 
perpLines = zeros(3, size(perp_line_indices,2));

for ii = 1:size(perp_line_indices,2)
    % compute already transformed lines
    perpLines(:, ii) =  H_r_aff.' \ getLine(lines(perp_line_indices(ii)).point1, lines(perp_line_indices(ii)).point2);
end


%% compute H through linear reg starting from affine

ls = [];
ms = [];
index = 1;
for ii = 1:2:size(perpLines,2)
    ls(:, index) = perpLines(:, ii);
    ms(:, index) = perpLines(:, ii+1);
    index = index +1;
end
H = getH_from_affine(ls,ms);

%% Transform

tform = projective2d(H.');
outputImage = imwarp(outputImage_aff, tform);
figure();
imshow(outputImage);

%% from original
H_r = H * H_r_aff 
tform = projective2d(H_r.');
outputImage = imwarp(im_rgb, tform);
figure();
imshow(outputImage);
hold on

%% Measure of metric properties (relative orientation)
% in order to determine the relative position between the vertical faces
% it's possible to measure the cosine of the angle between the two longest
% line in each face. (e.g. between line 1 and 4
line_indices_for_theta = [1, 4, 2, 4, 1, 39, 18, 4, 18, 39, 18, 24, 2, 24, 2, 3, 18, 3, 1, 3];

% collect estimates of theta
theta_est = zeros(1,size(line_indices_for_theta,2)/2)

index = 1
for ii = 1:2:size(line_indices_for_theta,2)
    l1 = getLine(lines(line_indices_for_theta(ii)).point1, lines(line_indices_for_theta(ii)).point2);
    l2 = getLine(lines(line_indices_for_theta(ii+1)).point1, lines(line_indices_for_theta(ii+1)).point2);

    % transform both lines according to the overall transformation
    l1 = H_r.' \ l1;
    l2 = H_r.' \ l2;
    l1 = l1(1:2,:);
    l2 = l2(1:2,:);

    % measure the cosine between line 1 and 4
    cos_theta = (l1.' * l2)/(norm(l1,2)*norm(l2,2));
    theta_est(index) = acosd(cos_theta);
    
    index = index + 1;
end

theta = mean(theta_est)

%% Measure of metric properties (relative position)

% measure longside of horizontal face and shortside for homography
% estimation

% get useful lines
l_left = H_r.' \ getLine(lines(1).point1, lines(1).point2);
l_up = H_r.' \ getLine(lines(55).point1, lines(55).point2);
l_down = H_r.' \ getLine(lines(15).point1, lines(15).point2);
l_right = H_r.' \ getLine(lines(18).point1, lines(18).point2);

% upper left point and down left point
x_ul_left = cross(l_left,l_up);
x_dl_left = cross(l_left,l_down);

% upper right point and down right point
x_ur_left = cross(l_right,l_up);
x_dr_left = cross(l_down,l_right);

% normalization
x_ul_left = x_ul_left ./ x_ul_left(3,1); 
x_dl_left = x_dl_left ./ x_dl_left(3,1);
x_ur_left = x_ur_left./ x_ur_left(3,1);
x_dr_left = x_dr_left./ x_dr_left(3,1);

% length of the longside of horizontal face using image measure
length_longside1 = norm(x_ul_left - x_dl_left,2);
length_longside2 = norm(x_ur_left - x_dr_left,2);

% do the average
length_longside = (length_longside1 + length_longside2) / 2;

% measure the length of the short side on the left part (needed for
% homography estimation
length_shortside1 = norm(x_ul_left - x_ur_left,2);
length_shortside2 = norm(x_dl_left - x_dr_left,2);

% do the average
length_shortside = (length_shortside1 + length_shortside2) / 2;

% calculate aspect ratio
aspect_ratio_left = length_longside/length_shortside;

%% Do the same as before for the right face (needed for homography estimation)

% get useful lines of right face
l_left_rf = H_r.' \ getLine(lines(39).point1, lines(39).point2);
l_up_rf = H_r.' \ getLine(lines(3).point1, lines(3).point2);
l_down_rf = H_r.' \ getLine(lines(62).point1, lines(62).point2);
l_right_rf = H_r.' \ getLine(lines(24).point1, lines(24).point2);

% upper left point and down left point
x_ul_right = cross(l_left_rf,l_up_rf);
x_dl_right = cross(l_left_rf,l_down_rf);

% upper right point and down right point
x_ur_right = cross(l_right_rf,l_up_rf);
x_dr_right = cross(l_down_rf,l_right_rf);

% normalization
x_ul_right = x_ul_right ./ x_ul_right(3,1); 
x_dl_right = x_dl_right ./ x_dl_right(3,1);
x_ur_right = x_ur_right./ x_ur_right(3,1);
x_dr_right = x_dr_right./ x_dr_right(3,1);

% length of the longside of horizontal face using image measure
length_longside1_r = norm(x_ul_right - x_dl_right,2);
length_longside2_r = norm(x_ur_right - x_dr_right,2);

% do the average
length_longside_r = (length_longside1_r + length_longside2_r) / 2;

% measure the length of the short side on the left part (needed for
% homography estimation
length_shortside1_r = norm(x_ul_right - x_ur_right,2);
length_shortside2_r = norm(x_dl_right - x_dr_right,2);

% do the average
length_shortside_r = (length_shortside1_r + length_shortside2_r) / 2;

% calculate aspect ratio
aspect_ratio_right = length_longside_r/length_shortside_r;

%% get the origin of the other face (intersection between 62 and 39
l62 = H_r.' \ getLine(lines(62).point1, lines(62).point2);
l39 = H_r.' \ getLine(lines(39).point1, lines(39).point2);

x_origin2 = cross(l62,l39);
x_origin2 = x_origin2 ./ x_origin2(3,1) 

x_origin1 = x_dl_left;

% get distance in the image
imaged_distance = norm(x_origin2 - x_origin1,2);

% relative position in mm
relative_position = imaged_distance * 243 / length_longside 

% relative coordinates (in world measures)
relative_coordinates = (x_origin2 - x_origin1).* 243 ./ length_longside

%% 2.2 Using also the images of vertical lines, calibrate the camera 
% (i.e., determine the calibration matrix K) assuming it is zero-skew 
% (but not assuming it is natural).
% use the image of vertical lines and intersect the vanishing points with
% the line at inf on the horizontal plane, the two vp on the vertical
% faces,
% use also the ortogonality of vp on the vertical faces

% find [l_inf]
% use l_inf_prime for two constraints
H_scaling = diag([1/4128, 1/4128, 1]);
l_infs = [H_scaling.' \ l_inf_prime];

% compute vanishing point of vertical direction of vertical planes
parallel_lines_vertical = [6,9,5,12,26,60,35,66,30,23,21,49];

% vertical lines of left vertical face
Ls = getLineMatrix(lines, parallel_lines_vertical);

% get vertical vanishing point
vp = getVp(Ls).*[1/size(im_rgb,2), 1/size(im_rgb,2), 1].';

% get horizontal vanishing point on each vertical face
vp_horiz_1 = cross(getLine(lines(15).point1, lines(15).point2), getLine(lines(51).point1, lines(51).point2));
vp_horiz_2 = cross(getLine(lines(62).point1, lines(62).point2), getLine(lines(34).point1, lines(34).point2));

vp_horiz_1 = vp_horiz_1./vp_horiz_1(3,1);
vp_horiz_1 = vp_horiz_1 .*[1/size(im_rgb,2), 1/size(im_rgb,2), 1].'

vp_horiz_2 = vp_horiz_2 ./ vp_horiz_2(3,1);
vp_horiz_2 = vp_horiz_2 .*[1/size(im_rgb,2), 1/size(im_rgb,2), 1].'

% get the vanishing points of the ground 
parallel_lines_ground = [10,16,19,50];
Ls_ground = getLineMatrix(lines, parallel_lines_ground);
vp_ground = getVp(Ls_ground).*[1/size(im_rgb,2), 1/size(im_rgb,2), 1].';
%% LET'S CALIBRATE!
IAC = getCalib_matrix(l_infs, vp, [vp_horiz_1, vp_horiz_2, vp_ground], [vp,vp,vp], H_scaling*inv(H_r));

% get the intrinsic parameter
alfa = sqrt(IAC(1,1))
u0 = -IAC(1,3)/(alfa^2)
v0 = -IAC(2,3)
fy = sqrt(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2))
fx = fy /alfa
K = [fx 0 u0; 0 fy v0; 0 0 1];
K = H_scaling \ K

%% Localization
% Refer to: 
% A Flexible New Technique for Camera Calibration
% by Zhengyou Zhang
% Here we first extract the Rotation of the plane wrt the camera frame, the
% we compute the rotation and translation of the camera wrt the plane.


% first fit the homography to restore real measure (possible since we know
% the aspect ratio
x_dl = [0 0];
x_ul = [0 243];
x_dr = [243/aspect_ratio_left 0];
x_ur = [243/aspect_ratio_left 243];

% get the same points in the original image
% get useful lines
l_left = getLine(lines(1).point1, lines(1).point2);
l_up = getLine(lines(55).point1, lines(55).point2);
l_down = getLine(lines(15).point1, lines(15).point2);
l_right = getLine(lines(18).point1, lines(18).point2);

% upper left point and down left point
x_ul_left = cross(l_left,l_up);
x_dl_left = cross(l_left,l_down);

% upper right point and down right point
x_ur_left = cross(l_right,l_up);
x_dr_left = cross(l_down,l_right);

% normalization
x_ul_left = x_ul_left ./ x_ul_left(3,1); 
x_dl_left = x_dl_left ./ x_dl_left(3,1);
x_ur_left = x_ur_left./ x_ur_left(3,1);
x_dr_left = x_dr_left./ x_dr_left(3,1);

H_omog = fitgeotrans([x_ul; x_dl; x_ur; x_dr], [x_ul_left(1:2).'; x_dl_left(1:2).'; x_ur_left(1:2).'; x_dr_left(1:2).'], 'projective');
H_omog = H_omog.T.';

% extract columns
h1 = H_omog(:,1);
h2 = H_omog(:,2);
h3 = H_omog(:,3);

lambda = 1 / norm(K \ h1);

% r1 = K^-1 * h1 normalized
r1 = (K \ h1) * lambda;
r2 = (K \ h2) * lambda;
r3 = cross(r1,r2);

R = [r1, r2, r3];

% due to noise in the data R may be not a true rotation matrix.
% approximate it through svd, obtaining a orthogonal matrix
[U, ~, V] = svd(R);
R = U * V';

% Compute translation vector. This vector is the position of the plane wrt
% the reference frame of the camera.
T = (K \ (lambda * h3));

cameraRotation = R.';
% since T is expressed in the camera ref frame we want it in the plane
% reference frame, R.' is the rotation of the camera wrt the plane
cameraPosition = -R.'*T;


%% Display orientation and location from left

figure
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);
hold on
pcshow([[x_ul; x_dl; x_ur; x_dr], zeros(size([x_ul; x_dl; x_ur; x_dr],1), 1)], ...
    'VerticalAxisDir', 'up', 'MarkerSize', 100);
xlabel('X')
ylabel('Y')
zlabel('Z')

%% Localization wrt the other face

% Build the Rotation matrix of the right face wrt the left face. This is
% done with a rotation of theta (defined before)
% R0->2 = R0->1 R1->2 (overall rotation)
% t0->2 = t0->1+R0->1*t1->2
R_from_left_to_right = rotz(-theta);

R_from_cam_to_right_plane = R_from_left_to_right*R;

%add 0 as z
relative_coordinates_3d = relative_coordinates;
relative_coordinates_3d(3,1) = 0;

T_from_cam_to_right_plane = T + R*relative_coordinates_3d;

cameraPos_wrt_right = -R_from_cam_to_right_plane.'*T_from_cam_to_right_plane;
camera_orientation_wrt_right = R_from_cam_to_right_plane.';

%% Display orientation and location from right
left_square = [[x_ul; x_dl; x_ur; x_dr], zeros(size([x_ul; x_dl; x_ur; x_dr],1), 1)];

% first add the translation, then rotate
%relative_coordinates_plot = [relative_coordinates_3d(2,1); relative_coordinates_3d(1,1); 0];
right_square = left_square %+ relative_coordinates_3d.';

right_square = [R_from_left_to_right*right_square(1,:).', R_from_left_to_right*right_square(2,:).' , R_from_left_to_right*right_square(3,:).', R_from_left_to_right*right_square(4,:).'];
% add traslation
right_square = right_square - relative_coordinates_3d;
right_square = right_square.';
figure
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);
hold on
pcshow(left_square, 'VerticalAxisDir', 'up', 'MarkerSize', 100);
hold on
pcshow(right_square, 'VerticalAxisDir', 'up', 'MarkerSize', 200);
xlabel('X')
ylabel('Y')
zlabel('Z')
