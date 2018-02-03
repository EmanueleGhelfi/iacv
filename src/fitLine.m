function line = fitLine(points,perp_distance)
%FITLINE Returns the line common to all points using LSA
%   points is a matrix 3*n containing all points
% if perp distance is true minimize the perpendicular distance (total LS), otherwise
% minimize the vertical distance (normal LS).
% the model is l_t x = 0 since the point belongs to the line

X = []; % should be nx2 matrix (n is points size 2) 
Y = []; % should be n matrix of -x3 elements

if ~perp_distance
    for ii = 1:size(points,2)
        % first get the point
        p = points(:,ii);

        % put the first two components inside x
        X(ii, :) = [p(1), p(2)];
        Y(ii, :) = -p(3);

    end
    W = (X.'*X)\(X.'*Y);
    line = [W(1,1), W(2,1), 1].';
else
    % point normalization
    points = points ./ points(3,:);
    points = points([1,2],:);
    points = points.';
    
    % subtract the mean in every direction
    A = points - mean(points);
    
    % get the smallest eigenvector of A'A
    [V,~] = eigs(A.'*A,1,'SM');
    a = V(1);
    b = V(2);
    
    % rho = a * x_mean + b * y_mean
    rho = [a b] * mean(points).';
    
    % line is rho = a x + b y
    line = [a b -rho].';
    
    % line normalization
    line = line ./ line(3,1);
end
