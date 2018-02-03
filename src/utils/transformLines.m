function lines = transformLines(H,ls)
%TRANSFORMLINES Transform all lines in ls using the matrix H
%   ls is a matrix 3 x n where n is the number of lines
%   H is the matrix to be used.
lines = zeros(size(ls));
for ii = 1:size(ls,2)
    % compute transformed lines
    lines(:, ii) =  H.' \ ls(:,ii);
end

