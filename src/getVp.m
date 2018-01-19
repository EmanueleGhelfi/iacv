function VP = getVp(ls)
%GETVP Returns the vanishing point common to all lines using LSA
%   lines is a matrix 3*n containing all lines having a common direction

X = []; % should be nx2 matrix (n is ls size 2) 
Y = []; % should be n matrix of -x3 elements

index = 1;
% computes vanishing points for each pair of lines
for ii = 1:size(ls,2)
    % first get the line
    l = ls(:,ii);
    
    % put the first two components inside x
    X(index, :) = [l(1), l(2)];
    Y(index, :) = -l(3);
    
    index = index + 1;
end
W = (X.'*X)\(X.'*Y);
VP = [W(1,1), W(2,1), 1].';



