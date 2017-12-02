%% clear all
clc
clear all

%% Open the image
im_rgb = imread('Input image.jpeg');
figure, imshow(im_rgb);

%% First convert the image in greyscale
im_grey = rgb2gray(im_rgb);
figure, imshow(im_grey);

%% Edge detection
% sobel, canny, prewitt, roberts
% BW2 = edge(im_grey,'canny');
[BW2, th] = edge(im_grey,'canny');
figure;
imshow(BW2)
title('Canny Filter');
th
%% Change Canny Threshold
% try changing threshold
th2 = th.*[1.2,2.5];
BW2 = edge(im_grey,'canny', th2);
figure, imshow(BW2);

%% Line Detection
% Hough parameters
minLineLength = 40;
fillGap = 40;
numPeaks = 40;

[H,theta,rho] = hough(BW2);

%% Display hough space
figure
imshow(imadjust(rescale(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)
%%
% find peaks in hough transform
P = houghpeaks(H,numPeaks,'threshold',ceil(0.2*max(H(:))));

% find lines using houghlines
lines = houghlines(BW2,theta,rho,P,'FillGap',fillGap,'MinLength',minLineLength);

%% plot lines
figure, imshow(im_rgb), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   text(xy(1,1),xy(1,2), num2str(k), 'Color', 'red')
end