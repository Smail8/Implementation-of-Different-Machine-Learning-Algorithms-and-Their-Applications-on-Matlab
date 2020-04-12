function [rimg] = reconstruct_image(labels,centroids,imgSize)
%RECONSTRUCT_IMAGE Reconstruct the image given the labels and the centroids
%
%   input -----------------------------------------------------------------
%   
%       o labels: The labels of the corresponding centroid for each pixel
%       o centroids: All the centroids and their RGB color value
%       o imgSize: Size of the original image for reconstruction
%
%   output ----------------------------------------------------------------
%
%       o rimg : The reconstructed image

% ADD CODE HERE: Reconstruct the image based on the labels on the centroids
% HINT: Apply the two steps you have used to reshape in the opposite order 
% if necessary
height = imgSize(1);
width = imgSize(2);
rimg = zeros(height, width, 3);
reshaped_labels = reshape(labels, height, width);
for i=1:height
    for j=1:width
        for k=1:3
            rimg(i, j, k) = centroids(k, reshaped_labels(i,j));
        end
    end
end


% END CODE
end
