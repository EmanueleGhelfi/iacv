function pairs = createLinePairs(l)
%CREATE_PAIRS Given a set of lines creates the matrix containing all
%possible pair of line
%   l1 is a matrix containing lines to be paired
pairs = [];
for ii = 1:size(l,2)
    for jj = (ii+1):size(l,2)
        pairs = [pairs l(:,ii) l(:,jj)];    
    end
end

