function H = lin_reg_h(ls,ms)
%LIN_REG Computes H through linear regression
%   ls is a matrix containing l lines
%   ms is a matrix containig m lines
%   ms and ls are orthogonal to the line in the same position

X = []; % should be nxm matrix (n is ls size 2, m is 5)
Y = []; % should be n matrix of l3m3 elements
for ii = 1:size(ls,2)
    % first compute the element of x
    li = ls(:,ii);
    mi = ms(:,ii);
    l1 = li(1,1);
    l2 = li(2,1);
    l3 = li(3,1);
    m1 = mi(1,1);
    m2 = mi(2,1);
    m3 = mi(3,1);
 
    X(ii,:) = [l1*m1, l1*m2+l2*m1, l2*m2, l1*m3+l3*m1, l2*m3+l3*m2];
    Y(ii,1) = -l3*m3;
end
X
Y
W = inv(X.'*X)*X.'*Y
C_star_prime = [W(1,1) W(2,1) W(3,1); W(2,1) W(4,1) W(5,1); W(3,1) W(5,1) 1];
C_star_prime
det(C_star_prime)
%C_star_prime = C_star_prime ./1e7

% decomposition
[U,S,V] = svd(C_star_prime)

H_scaling = diag([1/4128, 1/4128, 1])
U = H_scaling*U;
U = U * sqrt(diag([S(1,1), S(2,2), 1]))
H = U.';

