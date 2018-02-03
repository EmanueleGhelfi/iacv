function VP = getVp(ls)
%GETVP Returns the vanishing point common to all lines using LSA
%   lines is a matrix 3*n containing all lines having a common direction
% given lines the vanishing points obeys the law: l_t vp = 0
% considering the last element of vp =1 => ax + by = -c
% where the line is [a b c]'.

X = []; % should be nx2 matrix (n is ls size 2) 
Y = []; % should be n matrix of -x3 elements

for ii = 1:size(ls,2)
    % first get the line
    l = ls(:,ii);
    
    % put the first two components inside x
    X(ii, :) = [l(1), l(2)];
    Y(ii, :) = -l(3);
    
end
W = (X.'*X)\(X.'*Y);
VP = [W(1,1), W(2,1), 1].';



