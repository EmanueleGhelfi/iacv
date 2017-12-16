function lines = hough_lines(im, minLength, fillGap, numPeaks, angles, nHoodSize)
    % minLegth: minLength param of hough transform
    % fillGap:  fillGap param of hough transform
    % numPeaks: same as before
    % angles: angles found by this hough transform
    % im is the image
    % returns the line found by hough

    [H,theta,rho] = hough(im,'RhoResolution', 1, 'Theta', angles);
    % find peaks in hough transform
    P = houghpeaks(H,numPeaks,'threshold',ceil(0.1*max(H(:))), 'NHoodSize', nHoodSize);
    %P = houghpeaks(H,numPeaks,'threshold',ceil(0.1*max(H(:))));

    % find lines using houghlines
    lines = houghlines(im,theta,rho,P,'FillGap',fillGap,'MinLength',minLength);

end
