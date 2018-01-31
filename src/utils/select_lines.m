function [line_ind, lines_out] = select_lines(lines,img, text, auto_selection, line_group)
%SELECT_LINES Make the user select lines, text is displayed over the image
%in order to guide the user through line selection phase
%   lines: list of structs, like the one returned by hough lines
%   text: is displayed over the image
%   img: img readed
%   lines_out: output, lines selected by the user
% distance point line = |ax0 + by0 + c| / sqrt(a^2 + b^2)
% closest point to the line
%x={\frac  {b(bx_{0}-ay_{0})-ac}{a^{2}+b^{2}}}{\text{ and }}y={\frac  {a(-bx_{0}+ay_{0})-bc}{a^{2}+b^{2}}}.

% global variables used in other functions
global LINES_LONG_LEFT LINES_LONG_RIGHT LINES_SHORT_LEFT ...
       LINES_SHORT_RIGHT LINES_GROUND LINES_VERTICAL ...
       LINES_LEFT_ORDERED LINES_RIGHT_TWO ...
       LINES_VERTICAL_LEFT LINES_VERTICAL_RIGHT

   
% get lines for selection
% all the indices
indices = 1:length(lines);
L = getLineMatrix(lines, indices);

if (auto_selection)
    switch line_group
        case LINES_LONG_LEFT
            line_indices = [14 1 2 18];
        case LINES_LONG_RIGHT
            line_indices = [39 4 24 13 65 72];
        case LINES_SHORT_LEFT
            line_indices = [15 31 55];
        case LINES_SHORT_RIGHT
            line_indices = [62 33 68 76 71];
        case LINES_GROUND
            line_indices = [10 16 19 75 50];
        case LINES_VERTICAL
            line_indices = [26 6 9 5 12 45 35 66 30 49];
        case LINES_LEFT_ORDERED
            line_indices = [1 55 15 18];
        case LINES_RIGHT_TWO
            line_indices = [72 76];
        case LINES_VERTICAL_LEFT
            line_indices = [6 15 74 12];
        case LINES_VERTICAL_RIGHT
            line_indices = [35 76 73 30];
    end
else
    plot_lines(lines, img);
    title(text);

    % get the points selected by the user
    [x, y] = getpts;

    % close the figure
    close

    % array of indices
    line_indices = zeros(1,length(x));

    % for each point select the nearest line in lines
    for ii = 1:length(x)

        min_dist = 1000000;
        index = 0;

        % search the line with min distance
        for jj = 1:size(L,2)

            % get the line
            line = L(:,jj);

            a = line(1);
            b = line(2);
            c = line(3);

            % get the closest point to the line
            x_0 = (b * (b * x(ii) - a*y(ii)) - a*c) / (a^2 + b^2);
            y_0 = (a * (- b * x(ii) + a * y(ii) ) - b * c)/(a^2 + b^2);

            dist = abs(line(1)*x(ii) + line(2) * y(ii) + line(3)) / sqrt(line(1)^2 + line(2)^2);

            if (x_0 <= max(lines(jj).point1(1), lines(jj).point2(1))) ...
                && (x_0 >= min(lines(jj).point1(1), lines(jj).point2(1))) ...
                && (y_0 <= max(lines(jj).point1(2), lines(jj).point2(2))) ...
                && (y_0 >= min(lines(jj).point1(2), lines(jj).point2(2)))

                if dist <= min_dist
                    min_dist = dist;
                    index = jj;
                end
            end

        end

        % save index in line indices
        line_indices(ii) = index;
    end
end

lines_out = L(:,line_indices);
line_ind = line_indices;

end

