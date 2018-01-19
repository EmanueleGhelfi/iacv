function l_inf_prime = getL_inf_prime(ls)
%LIN_REG Computes l_inf_prime through least square approx
%   ls is a matrix containing l lines (pair of parallel lines)
% for each couple of lines we compute the intersection point,
% the line at infinite is the line minimizing the distance from these
% vanishing points.

X = []; % should be nx2 matrix (n is ls size 2) 
Y = []; % should be n matrix of -x3 elements

index = 1;
% computes vanishing points for each pair of lines
for ii = 1:2:(size(ls,2)-1)
    % first get the lines
    l1 = ls(:,ii);
    l2 = ls(:,ii+1);
    
    % compute vanishing point (intersection of parallel lines)
    vp = cross(l1,l2);
    
    % put the first two components inside x
    X(index, :) = [vp(1), vp(2)];
    Y(index, :) = -vp(3);
    
    index = index + 1;
end
W = (X.'*X)\(X.'*Y);
l_inf_prime = [W(1,1), W(2,1), 1].';

