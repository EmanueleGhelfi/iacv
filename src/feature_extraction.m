%% clear all
clc
clear all
debug = false; % do not diplay a thousand of images
%% Use Canny ( I do not remember if with or without contrast adjustment)
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
line_indices = [1, 2, 1, 18, 18, 2, 55, 31, 55, 15, 31, 15, 4, 24, 39, 4, 39, 24, 13, 24];

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
perp_line_indices = [1, 31, 31, 2, 4, 33, 24, 68, 67, 16, 31, 3, 2, 55, 68, 3, 39, 33, 62, 39];

% in this we save perpendicular lines 
perpLines = zeros(3, size(perp_line_indices,2));

for ii = 1:size(perp_line_indices,2)
    % compute already transformed lines
    perpLines(:, ii) =  H_r_aff.' \ getLine(lines(perp_line_indices(ii)).point1, lines(perp_line_indices(ii)).point2);
end

%% Shape reconstruction
% x*cos(theta) + y*sin(theta) - rho = 0
% compute lines
l1 = getLine(lines(1).point1, lines(1).point2);
l2 = getLine(lines(31).point1, lines(31).point2);
l3 = getLine(lines(31).point1, lines(31).point2);
l4 = getLine(lines(2).point1, lines(2).point2);
l5 = getLine(lines(4).point1, lines(4).point2);
l6 = getLine(lines(33).point1, lines(33).point2);
l7 = getLine(lines(24).point1, lines(24).point2);
l8 = getLine(lines(68).point1, lines(68).point2);
l9 = getLine(lines(67).point1, lines(67).point2);
l10 = getLine(lines(16).point1, lines(16).point2);
l11 = getLine(lines(31).point1, lines(31).point2);
l12 = getLine(lines(3).point1, lines(3).point2);
l13 = getLine(lines(2).point1, lines(2).point2);
l14 = getLine(lines(55).point1, lines(55).point2);
l15 = getLine(lines(68).point1, lines(68).point2);
l16 = getLine(lines(3).point1, lines(3).point2);
l17 = getLine(lines(39).point1, lines(39).point2);
l18 = getLine(lines(33).point1, lines(33).point2);
l19 = getLine(lines(62).point1, lines(62).point2);
l20 = getLine(lines(39).point1, lines(39).point2);

L = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20];
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

%% compute H
H = computeHFromOrtLines(L(:,1),L(:,2),L(:,3),L(:,4),L(:,5),L(:,6),L(:,7),L(:,8), L(:,9), L(:,10));

%% compute H through linear reg OLD
ls = [];
ms = [];
index = 1;
for ii = 1:2:size(L,2)
    ls(:, index) = L(:, ii);
    ms(:, index) = L(:, ii+1);
    index = index +1;
end
H = lin_reg_h(ls,ms);

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
tform = projective2d((H_r_aff * H).');
%tform = maketform('projective', H);
%tform.T = tform.T .* max(size(Im))
outputImage = imwarp(im_rgb, tform);
%Iout = imtransform(im_rgb, tform);
%outputImage = imwarp(im_rgb, tform);
figure();
imshow(outputImage);