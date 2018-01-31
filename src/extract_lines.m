function lines = extract_lines(im_rgb,debug, method)
%Extract Lines Perform preprocessing of the image including line extraction
%   Returns a set of line selected by the user after having performed canny
%   and hough transform
% im_rgb is the image file readed by im_read.
% debug: True display the images, False do not display.
% method: canny, roberts, sobel, log. Use canny.
%% First convert the image in greyscale
im_grey = double(rgb2gray(im_rgb))./255;
if debug
    figure, imshow(im_grey);
end

%% Edge detection with canny
% sobel, canny, prewitt, roberts
% BW2 = edge(im_grey,'canny');
if method == "canny"
    [BW2, th] = edge(im_grey,"canny");
    if debug
        figure;
        imshow(BW2)
        title('Canny Filter');
    end

    %% Change Canny Threshold
    % try changing threshold *1.5 is good
    th2 = th.*[0.2, 2];
    th3 = th.*[1.5, 2.5];
    
    BW2 = edge(im_grey,'canny', th2);
    BW3 = edge(im_grey,'canny', th3);
    if debug
        figure, imshow(BW2);
        
    end
end

%% with roberts
if method == "roberts"
    [BW2, th] = edge(im_grey,'roberts');
    figure;
    imshow(BW2)
    title('Roberts Filter');
    %% Change ROberts Threshold
    % try changing threshold
    th2 = th*0.7
    BW2 = edge(im_grey,'roberts', th2);
    figure, imshow(BW2);
end

%% Sobel
if method == "sobel"
    [BW2, th] = edge(im_grey,'Sobel', 0.035);
    figure;
    imshow(BW2)
    title('Sobel Filter');
    th
end

%% Log
if method == "log"
    [BW2, th] = edge(im_grey,'log', 0.0025);
    figure;
    imshow(BW2)
    title('Log Filter');
    th
end

%% Line Detection
% Hough parameters
% Use two sets of parameters for line extraction
minLineLength_vert = 170;
fillGap = 20;
numPeaks = 100;
NHoodSize = [111 81];
vertical_angles = -90:0.5:89.8;

% find lines vertical
[H,theta,rho] = hough(BW2,'RhoResolution', 1, 'Theta', vertical_angles);
% find peaks in hough transform
P = houghpeaks(H,numPeaks,'threshold',ceil(0.05*max(H(:))), 'NHoodSize', NHoodSize);

% find lines using houghlines
lines_1 = houghlines(BW2,theta,rho,P,'FillGap',fillGap,'MinLength',minLineLength_vert);

% Other params
minLineLength_vert = 160;
fillGap = 20;
numPeaks = 200;
NHoodSize = [101 51];
vertical_angles = -90:0.5:89.8;

% find lines vertical
[H,theta,rho] = hough(BW3,'RhoResolution', 1, 'Theta', vertical_angles);
% find peaks in hough transform
P = houghpeaks(H,numPeaks,'threshold',ceil(0.05*max(H(:))), 'NHoodSize', NHoodSize);

% find lines using houghlines
lines_2 = houghlines(BW3,theta,rho,P,'FillGap',fillGap,'MinLength',minLineLength_vert);

lines = [lines_1, lines_2(1,[80,89,86,77,17,32])];

%lines = houghlines(BW2,theta,rho,P);
if debug 
    plot_lines(lines, im_rgb)
end

