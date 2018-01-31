function plot_lines(lines,img)
%PLOT_LINES Plot lines over the image
%   lines is a vector of structs made of point1, point2 like the one
%   returned by the hough transform
% plot lines
figure, imshow(img), hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',1.5,'Color','green');

   % Plot beginnings and ends of lines
   %plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   %plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   text(xy(1,1),xy(1,2), num2str(k), 'Color', 'red')
   %text(xy(1,1),xy(1,2),strcat(num2str(xy(1,1)),",",num2str(xy(1,2))), 'Color', 'red')
end
end

