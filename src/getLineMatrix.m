function L = getLineMatrix(lines,indices)
%GETLINEMATRIX Returns a matrix with lines as columns
%   Detailed explanation goes here
% in this we save perpendicular lines 
Ls = zeros(3, size(indices,2));

for ii = 1:size(indices,2)
    % compute already transformed lines
    Ls(:, ii) = getLine(lines(indices(ii)).point1, lines(indices(ii)).point2);
end
L = Ls;
end

