function line = fitLine(points,perp_distance)
%GETVP Returns the line common to all points using LSA
%   points is a matrix 3*n containing all points

X = []; % should be nx2 matrix (n is points size 2) 
Y = []; % should be n matrix of -x3 elements

index = 1;
if ~perp_distance
    for ii = 1:size(points,2)
        % first get the point
        p = points(:,ii);

        % put the first two components inside x
        X(index, :) = [p(1), p(2)];
        Y(index, :) = -p(3);

        index = index + 1;
    end
    W = (X.'*X)\(X.'*Y);
    line = [W(1,1), W(2,1), 1].';
else
    % point normalization
    points = points ./ points(3,:);
    points = points([1,2],:);
    points = points.'
    A = points - mean(points);
    [U,~,V] = svd(A.'*A);
    % to do: go ahead 
end
