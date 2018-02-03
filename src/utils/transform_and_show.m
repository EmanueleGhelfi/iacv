function outputImage = transform_and_show(H,img, text)
%TRANSFORM_AND_SHOW Transforms the image using H and shows it
%   H is the transformation matrix to be applied to img
% text is the title of the image
% img is the original image

% create the tform object from H
tform = projective2d(H.');

% apply the transformation to img
outputImage = imwarp(img, tform);

% show
figure();
imshow(outputImage);
title(text);
end

