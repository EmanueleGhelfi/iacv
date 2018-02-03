%% clear all and instantiate variables
clc
clear all
debug = false; % do not diplay a thousand of images
auto_selection = true; %if true automatic line selection, if false manual line selection

% global variables used in other functions
global LINES_LONG_LEFT LINES_LONG_RIGHT LINES_SHORT_LEFT ...
       LINES_SHORT_RIGHT LINES_GROUND LINES_VERTICAL ...
       LINES_LEFT_ORDERED LINES_RIGHT_TWO LINES_VERTICAL_LEFT ...
       LINES_VERTICAL_RIGHT

% constants for line selection
LINES_LONG_LEFT = 1;
LINES_LONG_RIGHT = 2;
LINES_SHORT_LEFT = 3;
LINES_SHORT_RIGHT = 4;
LINES_GROUND = 5;
LINES_VERTICAL = 6;
LINES_LEFT_ORDERED = 7;
LINES_RIGHT_TWO = 8;
LINES_VERTICAL_LEFT = 9;
LINES_VERTICAL_RIGHT = 10;

% length of the longside of horizontal faces
LENGTH_LONGSIDE = 243;

%% Open the image
im_rgb = imread('Input_image.jpeg');

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
% l_inf -> l_inf (the line at infinity must be mapped to itself)
% H_r_aff = [1  0  0
%            0  1  0
%            l1 l2 l3] where the last row is l_inf'

% Extract parallel lines
% here we assume l1 and r1 are corresponding lines in the two faces
% and the same holds for l2 and r2
[line_ind_l1, lines_l1] = select_lines(lines,im_rgb,"Select the group of long parallel lines on the horizontal upper left face",auto_selection,LINES_LONG_LEFT);
[line_ind_l2, lines_l2] = select_lines(lines,im_rgb,"Select another group of parallel lines on the horizontal upper left face (orthogonal to the one selected before)", auto_selection, LINES_SHORT_LEFT);
[line_ind_r1, lines_r1] = select_lines(lines,im_rgb,"Select  the group of long parallel lines on the horizontal upper right face (corresponding lines selected in the first selection)", auto_selection, LINES_LONG_RIGHT);
[line_ind_r2, lines_r2] = select_lines(lines,im_rgb,"Select another group of parallel lines on the horizontal upper right face (orthogonal to the one selected before)", auto_selection, LINES_SHORT_RIGHT);
[line_ind_ground, lines_ground] = select_lines(lines,im_rgb,"Select a group of parallel lines on the ground", auto_selection, LINES_GROUND);

% plot selected lines
line_ind = [line_ind_l1, line_ind_l2, line_ind_r1, line_ind_r2, line_ind_ground];
plot_lines(lines(1,line_ind), im_rgb);

%% extract the line at infinite on the horizontal plane
% compute vanishing point of directions in the horizontal plane
% then fit the line through these points

% get vanishing points
vp_l1 = getVp(lines_l1);
vp_l2 = getVp(lines_l2);
vp_r1 = getVp(lines_r1);
vp_r2 = getVp(lines_r2);
vp_ground = getVp(lines_ground);

% fit the line through these points
l_inf_prime = fitLine([vp_l1 vp_l2 vp_r1 vp_r2 vp_ground],false);

%% compute H_r_aff (as before)

H_r_aff = [1 0 0; 0 1 0; l_inf_prime(1) l_inf_prime(2) l_inf_prime(3)];

% Transform the image and shows it
img_affine = transform_and_show(H_r_aff, im_rgb, "Affine rectification");

%% Metric rectification
% In order to perform metric rectification from an affine transformation we
% need perpendicular lines for constraints of the C_star_inf'

perpLines = [createLinePairsFromTwoSets(lines_l1, lines_l2), createLinePairsFromTwoSets(lines_r1, lines_r2)];

% transform lines according to H_r_aff since we need to start from an
% affinity
perpLines = transformLines(H_r_aff, perpLines);


%% compute H through linear reg starting from affine transformation

ls = [];
ms = [];
index = 1;
for ii = 1:2:size(perpLines,2)
    ls(:, index) = perpLines(:, ii);
    ms(:, index) = perpLines(:, ii+1);
    index = index + 1;
end

% fit the transformation from affinity to euclidean
H_a_e = getH_from_affine(ls,ms);

%% Transform from affinity

transform_and_show(H_a_e, img_affine, "Euclidean Reconstruction");

%% from original
% apply rotation of 180 degree along the x axis since the image is rotated
% around the x axis.
angle = 180;
R = rotx(deg2rad(180));

% calculate the composite transformation
% img -> affine -> euclidean -> rotation
% H_r is the transformation from the original image to the euclidean
% reconstruction.
H_r = R * H_a_e * H_r_aff;

out_e = transform_and_show(H_r, im_rgb, "Euclidean Reconstruction");


%% Measure of metric properties (relative orientation)
% in order to determine the relative position between the vertical faces
% it's possible to measure the cosine of the angle between the
% corresponding lines in each face and then do the average.
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
% theta is the angle from the right face and the left face (from the
% reference frame on the left face and the reference frame on the right
% face.
theta = mean(theta_est);

% Here we compute the angle from the reference frame on the left face and
% the reference frame of the image in order to express the position of the
% right face in the reference frame of the left face
% compute theta from left face to vertical direction
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
% Here we need to compute the imaged length of the longside of the left
% face since we know its real length (LENGTH_LONGSIDE) from the additional
% information provided.

% measure longside of left horizontal face and shortside for homography
% estimation
% select the 4 lines on the horizontal upper left face
[line_ind_left, lines_left_face] = select_lines(lines,im_rgb,"Select lines of the left horizontal face" ...
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

% calculate aspect ratio, this is the aspect ratio on the real world since
% now the image is a similarity.
aspect_ratio_left = length_longside_img/length_shortside_img;

% get the origin of the other face
% select the 2 lines on the horizontal upper right face
[line_ind_right, lines_right_face] = select_lines(lines,im_rgb,"Select the two lines on the right horizontal face" ...
                         + " such that the intersection is the origin of the face", auto_selection, LINES_RIGHT_TWO);
l_r1 = H_r.' \ lines_right_face(:,1);
l_r2 = H_r.' \ lines_right_face(:,2);

x_origin2 = cross(l_r1,l_r2);
x_origin2 = x_origin2 ./ x_origin2(3,1);

x_origin1 = x_dl_left;

% get distance in the image of the two reference frames
imaged_distance = norm(x_origin2 - x_origin1,2);

% relative distance in mm
relative_position = imaged_distance * LENGTH_LONGSIDE / length_longside_img;

% relative coordinates (in mm) in the reference frame of the image
relative_coordinates = (x_origin2 - x_origin1).* LENGTH_LONGSIDE ./ length_longside_img;


%% get the rotation of the left face wrt the image reference frame

% express the relative coordinates in the reference frame of the left face
R_from_img_to_left = rotz(deg2rad(theta_from_left_to_img));
R_last = roty(deg2rad(180));

% this is the relative position of the right face with respect to the left
% face
relative_pose_from_left_to_right = R_last*R_from_img_to_left*relative_coordinates;


%% 2.2 Using also the images of vertical lines, calibrate the camera 

% (i.e., determine the calibration matrix K) assuming it is zero-skew 
% (but not assuming it is natural).
% use the image of vertical lines and intersect the vertical vanishing point
% with the line at inf on the horizontal plane, use also the ortogonality 
% of vp on the vertical faces, use also the homography method
% Normalize all using H_scaling that brings one component in the range
% [0,1] while the other in the range [0, aspect_ratio].

% find [l_inf]
% use l_inf_prime for two constraints
% This is the scaling matrix that must be applied for normalization.
H_scaling = diag([1/IMG_MAX_SIZE, 1/IMG_MAX_SIZE, 1]);
l_infs = H_scaling.' \ l_inf_prime;

% compute vanishing point of vertical direction of vertical planes
[line_ind_vert, lines_vertical] = select_lines(lines,im_rgb,"Select vertical parallel lines", auto_selection, LINES_VERTICAL);

% plot all lines selected
if debug
    line_ind = [line_ind, line_ind_vert];
    plot_lines(lines(1,line_ind), im_rgb);
end

% get vertical vanishing point
vp_vertical = H_scaling * getVp(lines_vertical);

% get the vanishing point of the ground 
vp_ground = H_scaling * getVp(lines_ground);

%% LET'S CALIBRATE!
% there are 4 different ways for computing K using different constraints, 
% the last one uses all the constraints and maybe is better.

% Using L_inf, vertical vp and homography
IAC = get_IAC(l_infs, vp_vertical, [], [], H_scaling/H_r);

% get the intrinsic parameter before the denormalization
alfa = sqrt(IAC(1,1));
u0 = -IAC(1,3)/(alfa^2);
v0 = -IAC(2,3);
fy = sqrt(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2));
fx = fy /alfa;

% build K using the parametrization
K = [fx 0 u0; 0 fy v0; 0 0 1];

% denormalize K
K = H_scaling \ K

% get intrinsic after denormalization
fx = K(1,1)
fy = K(2,2)
u0 = K(1,3)
v0 = K(2,3)
alfa = fx/fy

%% Computation of K using only vp not normalized
vp_ground_de = getVp(lines_ground);
vp_vertical_de = getVp(lines_vertical);
IAC = get_IAC([],[], [vp_ground_de, vp_l1, vp_l2, vp_r1, vp_r2, vp_l1, vp_r1], [vp_vertical_de,vp_vertical_de,vp_vertical_de,vp_vertical_de,vp_vertical_de, vp_l2, vp_r2], []); 
alfa = sqrt(IAC(1,1))
u0 = -IAC(1,3)/(alfa^2)
v0 = -IAC(2,3)
fy = sqrt(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2))
fx = fy /alfa
K = [fx 0 u0; 0 fy v0; 0 0 1]

%% Computation of K using only Vp normalized

IAC = get_IAC([],[], [vp_ground, H_scaling * vp_l1, H_scaling * vp_l2, H_scaling * vp_r1,H_scaling * vp_r2, H_scaling *vp_l1, H_scaling *vp_r1], [vp_vertical,vp_vertical,vp_vertical,vp_vertical,vp_vertical,H_scaling * vp_l2,H_scaling * vp_r2], []); 
alfa = sqrt(IAC(1,1));
u0 = -IAC(1,3)/(alfa^2);
v0 = -IAC(2,3);
fy = sqrt(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2));
fx = fy /alfa;
K_de = [fx 0 u0; 0 fy v0; 0 0 1];

% denormalize K
K = H_scaling \ K_de

% get intrinsic after denormalization
fx = K(1,1)
fy = K(2,2)
u0 = K(1,3)
v0 = K(2,3)
alfa = fx/fy

%% The last way uses all the constraints
% are the better ways.
% Using L_inf and homography
IAC = get_IAC(l_infs, vp_vertical, [vp_ground, H_scaling * vp_l1, H_scaling * vp_l2, H_scaling * vp_r1,H_scaling * vp_r2, H_scaling *vp_l1, H_scaling *vp_r1], [vp_vertical,vp_vertical,vp_vertical,vp_vertical,vp_vertical,H_scaling * vp_l2,H_scaling * vp_r2], H_scaling/H_r);

% get the intrinsic parameter before the denormalization
alfa = sqrt(IAC(1,1));
u0 = -IAC(1,3)/(alfa^2);
v0 = -IAC(2,3);
fy = sqrt(IAC(3,3) - (alfa^2)*(u0^2) - (v0^2));
fx = fy /alfa;
K = [fx 0 u0; 0 fy v0; 0 0 1];
% denormalize K
K = H_scaling \ K

% get intrinsic after denormalization
fx = K(1,1)
fy = K(2,2)
u0 = K(1,3)
v0 = K(2,3)
alfa = fx/fy
%% Localization
% Refer to: 
% A Flexible New Technique for Camera Calibration
% by Zhengyou Zhang
% Here we first extract the Rotation of the (left) plane wrt the camera
% frame, then we compute the rotation and translation of the
% camera wrt the plane.


% first fit the homography to restore real measure (possible since we know
% the aspect ratio

% build up the rectangle of the left face using its real measure
x_dl = [0 0];
x_ul = [0 LENGTH_LONGSIDE];
x_dr = [LENGTH_LONGSIDE/aspect_ratio_left 0];
x_ur = [LENGTH_LONGSIDE/aspect_ratio_left LENGTH_LONGSIDE];

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
x_ur_left = x_ur_left./ x_ur_left(3,1);
x_dr_left = x_dr_left./ x_dr_left(3,1);

% fit the homography from scene to image 
H_omog = fitgeotrans([x_ul; x_dl; x_ur; x_dr], [x_ul_left(1:2).'; x_dl_left(1:2).'; x_ur_left(1:2).'; x_dr_left(1:2).'], 'projective');
H_omog = H_omog.T.';

% extract columns
h1 = H_omog(:,1);
h2 = H_omog(:,2);
h3 = H_omog(:,3);

% normalization factor.
lambda = 1 / norm(K \ h1);

% r1 = K^-1 * h1 normalized
r1 = (K \ h1) * lambda;
r2 = (K \ h2) * lambda;
r3 = cross(r1,r2);

% rotation of the world with respect to the camera (R cam -> world)
% where the world in this case is the left horizontal face
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


%% Display orientation and position from left horizontal face

figure
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);
hold on
pcshow([[x_ul; x_dl; x_ur; x_dr], zeros(size([x_ul; x_dl; x_ur; x_dr],1), 1)], ...
    'red','VerticalAxisDir', 'up', 'MarkerSize', 20);
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

% coordinates of center of right face with respect to the left face
relative_coordinates_3d = relative_pose_from_left_to_right;

% calculate displacement of the right plane wrt the camera
T_from_cam_to_right_plane = T + R*relative_coordinates_3d;

% calulcate the overall rotation and position of the camera with respect to
% right plane
cameraPos_wrt_right = -R_from_cam_to_right_plane.'*T_from_cam_to_right_plane;
camera_orientation_wrt_right = R_from_cam_to_right_plane.';

%% Display orientation and location from both faces

% 3d coordinates of the left face
left_face = [[x_ul; x_dl; x_ur; x_dr], zeros(size([x_ul; x_dl; x_ur; x_dr],1), 1)];

% in order to obtain the right face first rotate the left face then add 
% translation vector
right_face = left_face;

right_face = [R_from_left_to_right*right_face(1,:).', R_from_left_to_right*right_face(2,:).' , R_from_left_to_right*right_face(3,:).', R_from_left_to_right*right_face(4,:).'];

% add traslation
right_face = right_face + relative_pose_from_left_to_right;
right_face = right_face.';

% plot
figure
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);
hold on
pcshow(left_face,'red', 'VerticalAxisDir', 'up', 'MarkerSize', 20);
hold on
pcshow(right_face,'blue', 'VerticalAxisDir', 'up', 'MarkerSize', 20);
xlabel('X')
ylabel('Y')
zlabel('Z')

%% Display orientation and location from right face
figure
plotCamera('Location', cameraPos_wrt_right, 'Orientation', camera_orientation_wrt_right.', 'Size', 20);
hold on
% use the left face since it's the same
pcshow(left_face,'blue', 'VerticalAxisDir', 'up', 'MarkerSize', 20);
hold on
xlabel('X')
ylabel('Y')
zlabel('Z')

%% Vertical shape reconstruction
% a point on the left vertical face can be written as [x 0 z w] in the
% reference frame of the left horizontal face. So the reconstruction matrix
% from the image to its shape is simply [P_1 | P_3 | P_4]^(-1)

% reconstruction of the left face
% P = K * [R./lambda,T./lambda]; % projection matrix
P = K * [R,T]; % projection matrix
H_vert_l_sr = inv([P(:,1), P(:,3), P(:,4)]);
R_y = roty(deg2rad(180));

H_vert_l_sr = R_y * H_vert_l_sr;

out_lv = transform_and_show(H_vert_l_sr, im_rgb, "Shape reconstruction of" + ...
                                        "the left vertical face");
 
% reconstruction of the right face, same reasoning as before but using the
% reference frame on the right horizontal face.
% P2 = K * [R_from_cam_to_right_plane./lambda,T_from_cam_to_right_plane./lambda]; % projection matrix
P2 = K * [R_from_cam_to_right_plane,T_from_cam_to_right_plane]; % projection matrix
H_vert_r_sr = inv([P2(:,1), P2(:,3), P2(:,4)]);
R_y = roty(deg2rad(180));

% H_vert_sr is the matrix mapping the image to the reconstruction of the
% vertical face
H_vert_r_sr = R_y * H_vert_r_sr;

out_rv = transform_and_show(H_vert_r_sr, im_rgb, "Shape reconstruction of" + ...
                                        "the right vertical face");
 
                                  
%% Try to use P to reconstruct shape of horizontal faces.

H_oriz_sr = inv([P(:,1), P(:,2), P(:,4)]);
out_l = transform_and_show(H_oriz_sr, im_rgb, "Hor SR");
                                    
H_oriz_sr_r = inv([P2(:,1), P2(:,2), P2(:,4)]);
out = transform_and_show(H_oriz_sr_r, im_rgb, "Hor SR Right");

%% 3d coordinates for 3d shape reconstruction
% In this part the user should select lines on the vertical faces such that
% their intersections is the point to localize on the vertical faces
[line_ind_vert_l, lines_vertical_l] = select_lines(lines,im_rgb, ...
    "Select lines on the left face: left up down right",...
    auto_selection, LINES_VERTICAL_LEFT);
[line_ind_vert_r, lines_vertical_r] = select_lines(lines,im_rgb,...
    "Select lines on the right face: left up down right",...
    auto_selection, LINES_VERTICAL_RIGHT);

% plot all lines selected
if debug
    line_ind = [line_ind, line_ind_vert_l, line_ind_vert_r];
    plot_lines(lines(1,line_ind), im_rgb);
end

% get useful lines
l_v_left = lines_vertical_l(:,1);
l_v_up = lines_vertical_l(:,2);
l_v_down = lines_vertical_l(:,3);
l_v_right = lines_vertical_l(:,4);

% upper left point and down left point
x1 = cross(l_v_left,l_v_up);
x2 = cross(l_v_left,l_v_down);

% upper right point and down right point
x3 = cross(l_v_right,l_v_up);
x4 = cross(l_v_down,l_v_right);

% normalization
x1 = x1 ./ x1(3,1); 
x2 = x2 ./ x2(3,1);
x3 = x3./ x3(3,1);
x4 = x4./ x4(3,1);

points_l = [x1, x2, x3, x4];

% map points on the left face to their position
% points_lp stands for points_left_prime where prime means transformed
% through H_vert_l_sr
points_lp = H_vert_l_sr*points_l;
points_lp = points_lp ./ points_lp(3,:);
% remove last row of ones
points_lp = points_lp(1:2,:);

% change sign on the z axis
points_lp(2,:) = -points_lp(2,:);
points_lp = points_lp.';

% 3d points on the vertical face left
vert_points = [points_lp(:,1),zeros(size(points_lp,1),1), points_lp(:,2)];

% Localization of right vertical points

% get useful lines
l_v_left_r = lines_vertical_r(:,1);
l_v_up_r = lines_vertical_r(:,2);
l_v_down_r = lines_vertical_r(:,3);
l_v_right_r = lines_vertical_r(:,4);

% upper left point and down left point
x1 = cross(l_v_left_r,l_v_up_r);
x2 = cross(l_v_left_r,l_v_down_r);

% upper right point and down right point
x3 = cross(l_v_right_r,l_v_up_r);
x4 = cross(l_v_down_r,l_v_right_r);

% normalization
x1 = x1 ./ x1(3,1); 
x2 = x2 ./ x2(3,1);
x3 = x3./ x3(3,1);
x4 = x4./ x4(3,1);

points_r = [x1, x2, x3, x4];

% map points on the right vertical face to their position
% points_lp stands for points_left_prime where prime means transformed
points_rp = H_vert_r_sr*points_r;
points_rp = points_rp ./ points_rp(3,:);
points_rp = points_rp(1:2,:);

% change sign of the z axis.
points_rp(2,:) = -points_rp(2,:);
points_rp = points_rp.';
% add the 0 on the y axis
vert_points_p = [points_rp(:,1),zeros(size(points_rp,1),1), points_rp(:,2)];

% At this point we have vert_points_p that are the position of points on
% the reference frame of the right horizontal face. To obtain the position
% on the ref frame of the left face we have to add the translation and
% rotation needed.
% x_pi = t_p->pi' + R_pi->pi' x_pi' 
vert_points_p = relative_pose_from_left_to_right + (R_from_left_to_right * vert_points_p.');

vert_points = [vert_points; vert_points_p.'];

% plot all
figure
pcshow(left_face,'red', 'VerticalAxisDir', 'up', 'MarkerSize', 20);
hold on
pcshow(right_face,'blue', 'VerticalAxisDir', 'up', 'MarkerSize', 20);
pcshow(vert_points, 'green','VerticalAxisDir', 'up', 'MarkerSize', 20);
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);

%% 3d coordinates for 3d shaper rec
% in this part position of points on the vertical faces are determined,
% these points are extracted from the original image. Their positions is
% then determined using the matrices H_vert_l_sr and H_vert_r_sr and the
% information on the relative poisitions of the two reference frames (left
% and right horizontal faces).
% basically it's the same as before with the difference that the user can
% select any points on the vertical faces

% select point on the left vertical face to be mapped to its position
figure, imshow(im_rgb);
title("Select points on the left vertical face to be reconstructed");
[x, y] = getpts;
points_l = [x, y, ones(size(x,1),1)];
points_l = points_l.';

% map points on the left face to their position
points_p = H_vert_l_sr*points_l;

% Extract points on the right vertical face
title("Select points on the right vertical face to be reconstructed");
[x, y] = getpts;
points_r = [x, y, ones(size(x,1),1)];
points_r = points_r.';

points_r_p = H_vert_r_sr*points_r;
points_r_p = points_r_p ./ points_r_p(3,:);
points_r_p = points_r_p(1:2,:);

% change sign of the z axis.
points_r_p(2,:) = -points_r_p(2,:);
points_r_p = points_r_p.';
% add the 0 on the y axis
vert_points_p = [points_r_p(:,1),zeros(size(points_r_p,1),1), points_r_p(:,2)];

% At this point we have vert_points_p that are the position of points on
% the reference frame of the right horizontal face. To obtain the position
% on the ref frame of the left face we have to add the translation and
% rotation needed.
% x_pi = t_p->pi' + R_pi->pi' x_pi' 
vert_points_p = relative_pose_from_left_to_right + (R_from_left_to_right * vert_points_p.');


points_p = points_p ./ points_p(3,:);
% remove last row of ones
points_p = points_p(1:2,:);

% change sign on the z axis
points_p(2,:) = -points_p(2,:);
points_p = points_p.';

% plot all for 3d shape reconstruction
vert_points = [points_p(:,1),zeros(size(points_p,1),1), points_p(:,2); vert_points_p.'];

pcshow(left_face,'red', 'VerticalAxisDir', 'up', 'MarkerSize', 20);
hold on
pcshow(right_face,'blue', 'VerticalAxisDir', 'up', 'MarkerSize', 20);
pcshow(vert_points, 'green','VerticalAxisDir', 'up', 'MarkerSize', 20);
plotCamera('Location', cameraPosition, 'Orientation', cameraRotation.', 'Size', 20);

