function pairs = create_pairs(l)
%CREATE_PAIRS Given a set of indices creates the list containing all
%possible pairs
%   l is a list containing indices
pairs = [];
for ii = 1:size(l,2)
    for jj = (ii+1):size(l,2)
        pairs = [pairs l(ii) l(jj)];    
    end
end

