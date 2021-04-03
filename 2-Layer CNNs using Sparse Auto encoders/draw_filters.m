
function draw_filters(weights, height, width)
% draw weights

temp_range = quantile(weights(:),[0 0.001 0.005 0.05 0.25 0.50 0.75 0.95 0.995 0.999 1]);
fmax = max(abs(temp_range(:,[2,10])));
c_range = [-fmax, fmax];

figure;

imagesc(weights, c_range)
    set(gca, 'yDir', 'normal'); %colormap('gray')
    
% for i=1:25
%     subplot(5,5,i); imagesc(reshape(weights(:,i), height, width), c_range)
%     set(gca, 'yDir', 'normal'); %colormap('gray')
% end
% end