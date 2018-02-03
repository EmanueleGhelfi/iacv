function L = getLineMatrix(lines,indices)
%GETLINEMATRIX Returns a matrix with lines as columns
% lines are lines returned by hough lines
% indices are indices of lines to keep
Ls = zeros(3, size(indices,2));

for ii = 1:size(indices,2)
    % compute already transformed lines
    Ls(:, ii) = getLine(lines(indices(ii)).point1, lines(indices(ii)).point2);
end
L = Ls;
end

