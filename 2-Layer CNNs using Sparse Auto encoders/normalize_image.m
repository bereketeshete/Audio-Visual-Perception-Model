
function [train_video] = normalize_image(train_video, patch_width, patch_height)

% resize and normalize
mean=zeros(patch_width, patch_height);
num_frame = 0 ;
num_data = size(train_video, 1);
for idx=1:num_data
    image = train_video{idx,1};
    image_resize = zeros(patch_width, patch_height, size(image,3));
    for fidx = 1: size(image,3)
        frame = double(imresize(image(:,:,fidx), [patch_width, patch_height]))./255;
        mean = mean + frame;
        image_resize(:,:,fidx) = frame;
        num_frame = num_frame+1;
    end
    train_video{idx,1}=image_resize;
end
mean = mean/num_frame;

for idx=1:num_data
    train_video{idx,1}= bsxfun(@minus, train_video{idx,1}, mean);
end

end
