function H = getH_from_affine(ls,ms)
%GETH_FROM_AFFINE Computes H through least square approximation starting from an
%   affinity
% H is the reconstruction matrix that brings the affinity to an euclidean
% reconstruction
% The affine hypothesis is important since the last row is made of 0s.
%   ls is a matrix containing l lines
%   ms is a matrix containig m lines
%   ms and ls are orthogonal to the line in the same position
% if we know the angles in the real scene, the real lines obey this law:
% cos(theta) =              l1_t C_star_inf l2
%               -----------------------------------------------
%               sqrt(l1_t C_star_inf l1)  sqrt(l2_t C_star_inf l2)
%
% if the angles are ortogonal it's a linear equation
% Transforming using image elements:
% 0 = l1'_t C_star_inf_prime l2'
% C_star_inf_prime = [a b 0
%                     b 1 0
%                     0 0 0]
% 2 constraints are enough to determine C_star_inf
% here we use a least square approximation using all lines provided.

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

% closed form resolution of the system of equation.
W = (X.'*X)\(X.'*Y);
C_star_prime = [W(1,1) W(2,1) 0; W(2,1) 1 0; 0 0 0];

% build reconstruction matrix
% H = [K(1,1) K(1,2) 0; K(1,2) K(2,2) 0; 0 0 1]
[U, S, V] = svd(C_star_prime);
H = (U * diag([sqrt(S(1, 1)), sqrt(S(2, 2)), 1]));
H = inv(H);


