function pairs = create_pairs_from_two_lists(l1,l2)
%CREATE_PAIRS_FROM_TWO_LISTS Given two lists create the list containing all
%pairs such that one element is from the first and the second is from the
%second
%   l1 is a list containing indices
%   l2 is a list containing indices to be paired
pairs = [];
for ii = 1:size(l1,2)
    for jj = 1:size(l2,2)
        pairs = [pairs l1(ii) l2(jj)];    
    end
end

