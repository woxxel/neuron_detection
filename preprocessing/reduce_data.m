
function reduce_data(pathImages,path)
    
    disp(sprintf('calculating reduced images @ %s',path.reduced))
    
%     gcp;
    
    im = h5read(pathImages,'/DATA');
    width = size(im,2);
    height = size(im,1);
    bitDepth = 16;
    
%      tic
    max_im = zeros(height,width);
    ave_im = zeros(height,width);
    %% calculate MAX, MEDIAN and AVERAGE along one pixel
    for i = 1:height
      median_tmp = median(im(i,:,:),3);
      max_im(i,:) = max(im(i,:,:),[],3)-median_tmp;
      ave_im(i,:) = uint16(mean(im(i,:,:),3))-median_tmp;
    end
    
    %% normalize results to 16-bit depth
    max_im = max_im./2^bitDepth;
    ave_im = ave_im./2^bitDepth;
    
    save(path.reduced,'max_im','ave_im','-v7.3');
    
%      toc
end
