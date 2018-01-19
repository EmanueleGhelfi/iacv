function pairs = createLinePairsFromTwoSets(l1,l2)
%CREATELINEPAIRSFROMTWOSETS Given two matrices create the matrix containing all
%pairs such that one element is from the first and the second is from the
%second
%   l1 is a matrix containing lines
%   l2 is a matrix containing lines to be paired
pairs = [];
for ii = 1:size(l1,2)
    for jj = 1:size(l2,2)
        pairs = [pairs l1(:,ii) l2(:,jj)];    
    end
end
