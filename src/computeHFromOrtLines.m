function H = computeHFromOrtLines(l1,l2,l3,l4,l5,l6,l7,l8, l9, l10)
% lines should be ortogonal 
% if we know the angles in the real scene, the real lines obey this law:
% cos(theta) =              l1_t C_star_inf l2
%               -----------------------------------------------
%               sqrt(l1_t C_star_inf l1)  sqrt(l2_t C_star_inf l2)
%
% if the angles are ortogonal it's a linear equation
% Transforming using image elements:
% 0 = l1'_t C_star_inf_prime l2'
% we want to find C_start_inf_prime so we need 4 constraints
% 4 couples of orth lines
% The Use svd(C_star_inf_prime) = U S U'
% and H = U' 
% C_star_inf_prime is homog, symm and singular
% C_star_inf_prime = [ a b c
%                      b d e
%                      c e 1]
% singular -> det=0
%
    syms a b c d e
    C_star_inf_prime = [a b c; b d e; c e 1];
    
    % singular constraint (better to not use it since degree 3)
    %eq5 = det(C_star_inf_prime) == 0;
    
    % constraints on lines
    eq1 = l1.' * C_star_inf_prime * l2 == 0;
    % constraints on lines
    eq2 = l3.' * C_star_inf_prime * l4 == 0;
    % constraints on lines
    eq3 = l5.' * C_star_inf_prime * l6 == 0;
    % constraints on lines
    eq4 = l7.' * C_star_inf_prime * l8 == 0;
    % constraints on lines
    eq5 = l9.' * C_star_inf_prime * l10 == 0;
    
    % solve for H
    [A,B] = equationsToMatrix([eq1, eq2, eq3, eq4, eq5], [a, b, c, d, e]);
    X = linsolve(A,B)
    %X = solve([eq1, eq2, eq3, eq4, eq5], [a, b, c, d, e]);
    %X.a(1)
    %X.a(2)
    C_star_prime = double([X(1) X(2) X(3); X(2) X(4) X(5); X(3) X(5) 1]);
    %C_star_prime = double([X.a(1) X.b(1) X.c(1); X.b(1) X.d(1) X.e(1); X.c(1) X.e(1) 1]);
    C_star_prime
    C_star_prime = C_star_prime ./1e7
    det(C_star_prime)
    % decomposition
    [U,S,V] = svd(C_star_prime)
    
    %H_scaling = diag([1/4128, 1/4128, 1]);
    %U = U*H_scaling;
    % adj factor
    U = U * sqrt(diag([S(1,1), S(2,2), 1]))
    H = U.';

end