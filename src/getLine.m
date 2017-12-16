function line = getLine(x1,x2)
%GETLINE returns line in homogeneous coordinates
%  x1 and x2 are extreme points of the line
x1 = [x1(1), x1(2), 1];
x2 = [x2(1), x2(2), 1];
%x1 = [x1(1)/4128, x1(2)/4128, 1];
%x2 = [x2(1)/4128, x2(2)/4128, 1];
temp = cross(x1,x2);
line = (temp./temp(3)).';
end

