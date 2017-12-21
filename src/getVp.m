function VP = getVp(lines)
%GETVP Returns the vanishing point common to all lines doing an average
%   lines is a matrix 3*n containing all lines having a common direction

vps = [];
index = 1;
for ii = 1:size(lines,2)
    for jj = ii+1:size(lines,2)
        % first compute the element of x
        li = lines(:,ii);
        mi = lines(:,jj);
        vp = cross(li, mi);
        vp = vp./vp(3,1);
        vps(:, index) = vp
        index = index + 1;
    end
end

VP = mean(vps,2);

