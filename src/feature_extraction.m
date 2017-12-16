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
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

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
%      0  1  0
%      l1 l2 l3] where the last line is l_inf'

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

%% Plot intersection points
% lines are correct !!!!!

for ii = 1:2:length(perpLines)
    point = cross(perpLines(:,ii), perpLines(:,ii+1));
    point = point./point(3);
    plot(point(1),point(2), 'x','LineWidth',10,'Color','yellow');
end

%% scaling matrix for image normalization
H_scaling = diag([1/4186, 1/4186, 1]);

for ii = 1:size(L,2)
    L(:,ii) = inv(H_scaling.')*L(:,ii)
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

% measure longside of horizontal face

% get useful lines
l1 = H_r.' \ getLine(lines(1).point1, lines(1).point2);
l55 = H_r.' \ getLine(lines(55).point1, lines(55).point2);
l15 = H_r.' \ getLine(lines(15).point1, lines(15).point2);

% upper left point
x_1_55 = cross(l1,l55);
x_1_15 = cross(l1,l15);

% normalization
x_1_55 = x_1_55 ./ x_1_55(3,1) 
x_1_15 = x_1_15 ./ x_1_15(3,1) 

length_longside = norm(x_1_55 - x_1_15,2);

%% Test using the other side


%% get the origin of the other face (intersection between 62 and 39
l62 = H_r.' \ getLine(lines(62).point1, lines(62).point2);
l39 = H_r.' \ getLine(lines(39).point1, lines(39).point2);

x_origin2 = cross(l62,l39);
x_origin2 = x_origin2 ./ x_origin2(3,1) 

x_origin1 = x_1_15;

% get distance in the image
imaged_distance = norm(x_origin2 - x_origin1,2);

% relative position in mm
relative_position = imaged_distance * 243 / length_longside 

% relative coordinates (in world measures)
relative_coordinates = (x_origin2 - x_origin1).* 243 ./ length_longside
relative_coordinates(3,1) = 1










