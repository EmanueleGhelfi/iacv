function H = getH_from_affine(ls,ms)
%LIN_REG Computes H through least square approximation
%   ls is a matrix containing l lines
%   ms is a matrix containig m lines
%   ms and ls are orthogonal to the line in the same position

X = []; % should be nxm matrix (n is ls size 2, m is 2)
Y = []; % should be n matrix of -l2m2 elements
for ii = 1:size(ls,2)
    % first compute the element of x
    li = ls(:,ii);
    mi = ms(:,ii);
    l1 = li(1,1);
    l2 = li(2,1);
    m1 = mi(1,1);
    m2 = mi(2,1);
 
    X(ii,:) = [l1*m1, l1*m2+l2*m1];
    Y(ii,1) = -l2*m2;
end

W = (X.'*X)\(X.'*Y);
%S = [W(1,1) W(2,1); W(2,1) 1]; % = KK'
C_star_prime = [W(1,1) W(2,1) 0; W(2,1) 1 0; 0 0 0];

% decomposition using cholesky (not working)
%K = chol(S)

% build affine transformation
% H = [K(1,1) K(1,2) 0; K(1,2) K(2,2) 0; 0 0 1]
[U, S, V] = svd(C_star_prime);
H = (U * diag([sqrt(S(1, 1)), sqrt(S(2, 2)), 1]));
H = inv(H);

%[U, S, V] = svd(S)
%U = U*diag([1/sqrt(S(1, 1)), 1/sqrt(S(2, 2))])
%H = eye(3)
%H(1,1) = U(1,1);
%H(1,2) = U(1,2);
%H(2,2) = U(2,2);
%H(2,1) = U(2,1);


