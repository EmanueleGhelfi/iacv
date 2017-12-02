%% clear all
clc
clear all

%% Open the image
im_rgb = imread('Input image.jpeg');
figure, imshow(im);

%% First convert the image in greyscale
im_grey = rgb2gray(im_rgb);
figure, imshow(im_grey)