function outputImage = transform_and_show(H,img, text)
%TRANSFORM_AND_SHOW Transforms the image and shows it
%   H is the transformation matrix to be applied to img
% text is the title of the image
tform = projective2d(H.');
outputImage = imwarp(img, tform);
figure();
imshow(outputImage);
title(text);
end

