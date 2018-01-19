%% clear all and instantiate variables
clc
clear all
debug = false; % do not diplay a thousand of images
auto_selection = true; % automatic line selection

% global variables used in other functions
global LINES_LONG_LEFT LINES_LONG_RIGHT LINES_SHORT_LEFT ...
       LINES_SHORT_RIGHT LINES_GROUND LINES_VERTICAL ...
       LINES_LEFT_ORDERED LINES_RIGHT_TWO

% constants in the code
LINES_LONG_LEFT = 1;
LINES_LONG_RIGHT = 2;
LINES_SHORT_LEFT = 3;
LINES_SHORT_RIGHT = 4;
LINES_GROUND = 5;
LINES_VERTICAL = 6;
LINES_LEFT_ORDERED = 7;
LINES_RIGHT_TWO = 8;

% length of the longside of horizontal faces
LENGTH_LONGSIDE = 243;

%% Open the image
im_rgb = imread('Input image.jpeg');

IMG_MAX_SIZE = max(size(im_rgb));

if debug
    figure, imshow(im_rgb);
end

%% Line extraction
lines = extract_lines(im_rgb, debug, "canny");

% plot lines on the image
plot_lines(lines, im_rgb);

%% Stratification approach:
% - compute affine reconstruction
% l_inf -> l_inf
% H_r_aff = [1  0  0
%            0  1  0
%            l1 l2 l3] where the last line is l_inf'

% Extract parallel lines
% here we assume l1 and r1 are corresponding lines in the two faces
% and the same holds for l2 and r2
% if you want automatic extraction of lines decomment the line below and
% comment lines from 30 to 37
% line_indices = [1, 2, 1, 18, 18, 2, 55, 31, 55, 15, 31, 15, 4, 24, 39, 4, 39, 24, 13, 24, 50, 19, 19, 16, 68, 33, 3, 24, 3, 39];
lines_l1 = select_lines(lines,im_rgb,"Select the group of long parallel lines on the horizontal upper left face",auto_selection,LINES_LONG_LEFT);
lines_l2 = select_lines(lines,im_rgb,"Select another group of parallel lines on the horizontal upper left face (orthogonal to the one selected before)", auto_selection, LINES_SHORT_LEFT);
lines_r1 = select_lines(lines,im_rgb,"Select  the group of long parallel lines on the horizontal upper right face (corresponding lines selected in the first selection)", auto_selection, LINES_LONG_RIGHT);
lines_r2 = select_lines(lines,im_rgb,"Select another group of parallel lines on the horizontal upper right face (orthogonal to the one selected before)", auto_selection, LINES_SHORT_RIGHT);
lines_ground = select_lines(lines,im_rgb,"Select a group of parallel lines on the ground", auto_selection, LINES_GROUND);

% build up the list of pairs of parallel lines
%line_indices = [create_pairs(line_indices_l1), create_pairs(line_indices_l2), create_pairs(line_indices_r1), create_pairs(line_indices_r2), create_pairs(line_indices_ground)];

% in this we save pairs of parallel lines 
% parallelLines = getLineMatrix(lines,line_indices);
%parallelLines = [createLinePairs(lines_l1), createLinePairs(lines_l2), createLinePairs(lines_r1), createLinePairs(lines_r2), createLinePairs(lines_ground)]; 

%% extract the line at infinite on the horizontal plane
% older version, this version cosiders couple of parallel lines, finds its
% vp and then fit the lines through these points. Less accurate
%l_inf_prime = getL_inf_prime(parallelLines);

%% Another approach in order to find l_inf_prime:
% compute vanishing point of directions in the horizontal plane
% then fit the line through these points

vp_l1 = getVp(lines_l1);
vp_l2 = getVp(lines_l2);
vp_r1 = getVp(lines_r1);
vp_r2 = getVp(lines_r2);
vp_ground = getVp(lines_ground);

l_inf_prime = fitLine([vp_l1 vp_l2 vp_r1 vp_r2 vp_ground]);

%% compute H_r_aff (as before)

H_r_aff = [1 0 0; 0 1 0; l_inf_prime(1) l_inf_prime(2) l_inf_prime(3)];

% Transform the image
tform_r_aff = projective2d(H_r_aff.');
outputImage_aff = imwarp(im_rgb, tform_r_aff);
figure();
imshow(outputImage_aff);
title("Affine Rectification");

%% Metric rectification
% perpendicular line indices
% perp_line_indices = [1, 31, 31, 2, 4, 33, 24, 68, 67, 16, 31, 8, 2, 55, 68, 3, 39, 33, 62, 39, 67, 19, 55, 1, 55, 18, 31, 18, 24, 33, 67, 50];
%perp_line_indices = select_lines(lines,im_rgb,"Select pairs of perpendicular lines on the horizontal upper faces");
perpLines = [createLinePairsFromTwoSets(lines_l1, lines_l2), createLinePairsFromTwoSets(lines_r1, lines_r2)];

perpLines = transformLines(H_r_aff, perpLines);


%% compute H through linear reg starting from affine

ls = [];
ms = [];
index = 1;
for ii = 1:2:size(perpLines,2)
    ls(:, index) = perpLines(:, ii);
    ms(:, index) = perpLines(:, ii+1);
    index = index + 1;
end
H = getH_from_affine(ls,ms);

%% Transform

tform = projective2d(H.');
outputImage = imwarp(outputImage_aff, tform);
figure();
imshow(outputImage);

%% from original
% apply rotation of 180 degree along the x axis since the image is rotated
% around the x axis.
angle = 180;
R = rotx(deg2rad(180));

% calculate the composite transformation
H_r = R * H * H_r_aff;
tform = projective2d(H_r.');

% apply the transformation to the image
outputImage = imwarp(im_rgb, tform);
figure();
imshow(outputImage);
hold on

%% Measure of metric properties (relative orientation)
% in order to determine the relative position between the vertical faces
% it's possible to measure the cosine of the angle between the two longest
% line in each face. (e.g. between line 1 and 4)
%line_indices_for_theta = [1, 4, 2, 4, 1, 39, 18, 4, 18, 39, 18, 24, 2, 24, 2, 3, 18, 3, 1, 3];
%line_indices_for_theta = select_lines(lines,im_rgb,"Select pairs of lines on the horizontal upper faces in order to find the relative orientation");
lines_for_theta = [createLinePairsFromTwoSets(lines_l1, lines_r1), createLinePairsFromTwoSets(lines_l2, lines_r2)];

% transform lines according to H_r
lines_for_theta = transformLines(H_r, lines_for_theta);

% collect estimates of theta
theta_est = zeros(1,size(lines_for_theta,2)/2);

index = 1;
for ii = 1:2:size(lines_for_theta,2)
    l1 = lines_for_theta(:,ii);
    l2 = lines_for_theta(:,ii+1);
    
    % extract only the useful part for computing the angle
    l1 = l1(1:2,:); 
    l2 = l2(1:2,:);

    % measure the cosine between lines
    cos_theta = (l1.' * l2)/(norm(l1,2)*norm(l2,2));
    theta_est(index) = acosd(cos_theta);
    
    index = index + 1;
end

% do the average of the estimates
theta = mean(theta_est);

% compute theta from left face to vertical direction
%line_indices_for_theta = [1, 18, 14, 2];
%line_indices_for_theta_left = select_lines(lines,im_rgb,"Select the long parallel lines" ...
%                         + " on the horizontal upper left face for estimating its orientation");

lines_left_theta = transformLines(H_r, lines_l1);
                     
% collect estimates of theta
theta_left_est = zeros(1,size(lines_left_theta,2));

for ii = 1:size(lines_left_theta,2)
    
    % get the line of the left face
    l1 = lines_left_theta(:,ii);
    l2 = [1; 0; 0]; % vertical line in the image
    
    % get only the component of the direction
    l1 = l1(1:2,:);
    l2 = l2(1:2,:);

    % measure the cosine between line 1 and 4
    cos_theta = (l1.' * l2)/(norm(l1,2)*norm(l2,2));
    theta_left_est(ii) = acosd(cos_theta);
    
end

% this is the orientation of the left face wrt the image reference frame
theta_from_left_to_img = mean(theta_left_est) + 180;


%% Measure of metric properties (relative position)

% measure longside of horizontal face and shortside for homography
% estimation
% select the 4 lines on the horizontal upper left face
% line_indices_left_face = [1,55,15,18]
lines_left_face = select_lines(lines,im_rgb,"Select lines of the left horizontal face" ...
                         + " in the following order: left, up, down right", auto_selection, LINES_LEFT_ORDERED);

% get useful lines and transform them
l_left = H_r.' \ lines_left_face(:,1);
l_up = H_r.' \ lines_left_face(:,2);
l_down = H_r.' \ lines_left_face(:,3);
l_right = H_r.' \ lines_left_face(:,4);

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
length_longside_img = (length_longside1 + length_longside2) / 2;

% measure the length of the short side on the left part (needed for
% homography estimation
length_shortside1 = norm(x_ul_left - x_ur_left,2);
length_shortside2 = norm(x_dl_left - x_dr_left,2);

% do the average
length_shortside_img = (length_shortside1 + length_shortside2) / 2;

% calculate aspect ratio
aspect_ratio_left = length_longside_img/length_shortside_img;

%% Do the same as before for the right face (needed for homography estimation)
% OLD VERSION
% % get useful lines of right face
% l_left_rf = H_r.' \ getLine(lines(39).point1, lines(39).point2);
% l_up_rf = H_r.' \ getLine(lines(3).point1, lines(3).point2);
% l_down_rf = H_r.' \ getLine(lines(62).point1, lines(62).point2);
% l_right_rf = H_r.' \ getLine(lines(24).point1, lines(24).point2);
% 
% % upper left point and down left point
% x_ul_right = cross(l_left_rf,l_up_rf);
% x_dl_right = cross(l_left_rf,l_down_rf);
% 
% % upper right point and down right point
% x_ur_right = cross(l_right_rf,l_up_rf);
% x_dr_right = cross(l_down_rf,l_right_rf);
% 
% % normalization
% x_ul_right = x_ul_right ./ x_ul_right(3,1); 
% x_dl_right = x_dl_right ./ x_dl_right(3,1);
% x_ur_right = x_ur_right./ x_ur_right(3,1);
% x_dr_right = x_dr_right./ x_dr_right(3,1);
% 
% % length of the longside of horizontal face using image measure
% length_longside1_r = norm(x_ul_right - x_dl_right,2);
% length_longside2_r = norm(x_ur_right - x_dr_right,2);
% 
% % do the average
% length_longside_r = (length_longside1_r + length_longside2_r) / 2;
% 
% % measure the length of the short side on the left part (needed for
% % homography estimation
% length_shortside1_r = norm(x_ul_right - x_ur_right,2);
% length_shortside2_r = norm(x_dl_right - x_dr_right,2);
% 
% % do the average
% length_shortside_r = (length_shortside1_r + length_shortside2_r) / 2;
% 
% % calculate aspect ratio
% aspect_ratio_right = length_longside_r/length_shortside_r;

%% get the origin of the other face
% select the 2 lines on the horizontal upper right face
% line_indices_left_face = [62,39,15,18]
lines_right_face = select_lines(lines,im_rgb,"Select the two lines on the right horizontal face" ...
                         + " such that the intersection is the origin of the face", auto_selection, LINES_RIGHT_TWO);
l_r1 = H_r.' \ lines_right_face(:,1);
l_r2 = H_r.' \ lines_right_face(:,2);

x_origin2 = cross(l_r1,l_r2);
x_origin2 = x_origin2 ./ x_origin2(3,1);

x_origin1 = x_dl_left;

% get distance in the image
imaged_distance = norm(x_origin2 - x_origin1,2);

% relative position in mm
relative_position = imaged_distance * LENGTH_LONGSIDE / length_longside_img;

% relative coordinates (in mm)
relative_coordinates = (x_origin2 - x_origin1).* LENGTH_LONGSIDE ./ length_longside_img;

% change the y sign since the y axis goes down and we want it to go up
%relative_coordinates(2,1) = -relative_coordinates(2,1);

%% get the rotation of the left face wrt the image reference frame

% express the relative coordinates in the reference frame of the left face
R_from_img_to_left = rotz(deg2rad(theta_from_left_to_img));
R_last = roty(deg2rad(180));
relative_pose_from_left_to_right = R_last*R_from_img_to_left*relative_coordinates;


%% 2.2 Using also the images of vertical lines, calibrate the camera 

% (i.e., determine the calibration matrix K) assuming it is zero-skew 
% (but not assuming it is natural).
% use the image of vertical lines and intersect the vanishing points with
% the line at inf on the horizontal plane, the two vp on the vertical
% faces,
% use also the ortogonality of vp on the vertical faces

% find [l_inf]
% use l_inf_prime for two constraints
H_scaling = diag([1/IMG_MAX_SIZE, 1/IMG_MAX_SIZE, 1]);
l_infs = [H_scaling.' \ l_inf_prime];

% compute vanishing point of vertical direction of vertical planes
%parallel_lines_vertical = [6,9,5,12,26,60,35,66,30,23,21,49];
lines_vertical = select_lines(lines,im_rgb,"Select vertical parallel lines", auto_selection, LINES_VERTICAL);

% get vertical vanishing point
vp_vertical = H_scaling * getVp(lines_vertical); %.*[1/IMG_MAX_SIZE, 1/size(im_rgb,2), 1].';

% get horizontal vanishing point on each vertical face
% parallel_lines_horizontal_vert_face_l = [15 51]
% lines_horizontal_vert_face_l = select_lines(lines,im_rgb,"Select the two horizontal parallel lines on the left vertical face");
% vp_horiz_1 = cross(lines_horizontal_vert_face_l(:,1), lines_horizontal_vert_face_l(:,2));
% lines_horizontal_vert_face_r = select_lines(lines,im_rgb,"Select the two horizontal parallel lines on the right vertical face");
% vp_horiz_2 = cross(lines_horizontal_vert_face_r(:,1), lines_horizontal_vert_face_r(:,2));
% 
% % normalize data
% vp_horiz_1 = vp_horiz_1./vp_horiz_1(3,1);
% vp_horiz_1 = vp_horiz_1 .*[1/size(im_rgb,2), 1/size(im_rgb,2), 1].';
% 
% vp_horiz_2 = vp_horiz_2 ./ vp_horiz_2(3,1);
% vp_horiz_2 = vp_horiz_2 .*[1/size(im_rgb,2), 1/size(im_rgb,2), 1].';

% get the vanishing point of the ground 
vp_ground = H_scaling * getVp(lines_ground); %.*[1/size(im_rgb,2), 1/size(im_rgb,2), 1].';
%% LET'S CALIBRATE!
IAC = getCalib_matrix(l_infs, vp_vertical, vp_ground, vp_vertical, H_scaling*inv(H_r));

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
l_left = lines_left_face(:,1);
l_up = lines_left_face(:,2);
l_down = lines_left_face(:,3);
l_right = lines_left_face(:,4);

% upper left point and down left point
x_ul_left = cross(l_left,l_up);
x_dl_left = cross(l_left,l_down);

% upper right point and down right point
x_ur_left = cross(l_right,l_up);
x_dr_left = cross(l_down,l_right);

% normalization
x_ul_left = x_ul_left ./ x_ul_left(3,1); 
x_dl_left = x_dl_left ./ x_dl_left(3,1);
x_ur_left = x_ur_left ./ x_ur_left(3,1);
x_dr_left = x_dr_left ./ x_dr_left(3,1);

% fit the homography
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
R = U * V'

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
    'VerticalAxisDir', 'up', 'MarkerSize', 200);
xlabel('X')
ylabel('Y')
zlabel('Z')

%% Localization wrt the other face

% Build the Rotation matrix of the right face wrt the left face. This is
% done with a rotation of theta (defined before)
% R0->2 = R0->1 R1->2 (overall rotation)
% t0->2 = t0->1+R0->1*t1->2
R_from_left_to_right = rotz(-deg2rad(theta));

R_from_cam_to_right_plane = R*R_from_left_to_right;

relative_coordinates_3d = relative_coordinates;

% calculate displacement of the right plane wrt the camer
T_from_cam_to_right_plane = T + R*relative_coordinates_3d;

% calulcate the overall rotation of the right plane wrt the camera
cameraPos_wrt_right = -R_from_cam_to_right_plane.'*T_from_cam_to_right_plane;
camera_orientation_wrt_right = R_from_cam_to_right_plane.';

%% Display orientation and location from both faces
left_square = [[x_ul; x_dl; x_ur; x_dr], zeros(size([x_ul; x_dl; x_ur; x_dr],1), 1)];

% first add the translation, then rotate
%relative_coordinates_plot = [relative_coordinates_3d(2,1); relative_coordinates_3d(1,1); 0];
right_square = left_square; %+ relative_coordinates_3d.';

right_square = [R_from_left_to_right*right_square(1,:).', R_from_left_to_right*right_square(2,:).' , R_from_left_to_right*right_square(3,:).', R_from_left_to_right*right_square(4,:).'];

% add traslation
right_square = right_square + relative_pose_from_left_to_right;
right_square = right_square.';

% plot
figure
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);
hold on
pcshow(left_square, 'VerticalAxisDir', 'up', 'MarkerSize', 200);
hold on
pcshow(right_square, 'VerticalAxisDir', 'up', 'MarkerSize', 200);
xlabel('X')
ylabel('Y')
zlabel('Z')

%% Display orientation and location from right face
figure
plotCamera('Location', cameraPos_wrt_right, 'Orientation', camera_orientation_wrt_right.', 'Size', 20);
hold on
pcshow(left_square, 'VerticalAxisDir', 'up', 'MarkerSize', 200);
hold on
xlabel('X')
ylabel('Y')
zlabel('Z')

%% Vertical shape reconstruction

P = K * [R./lambda,T./lambda];
H_vert_sr = inv([P(:,1), P(:,3), P(:,4)]);
R_y = roty(deg2rad(180));
tform = projective2d((R_y*H_vert_sr).');

% apply the transformation to the image
outputImage = imwarp(im_rgb, tform);
figure();
imshow(outputImage);
hold on



