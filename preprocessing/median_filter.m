
function median_filter(pathImages,pathMedian,parameter)
    
    plt = true;
    
    disp(sprintf('median filtering @ %s with median filter size: %d,%d,%d',pathMedian,parameter.filtersize(1),parameter.filtersize(2),parameter.filtersize(3)))
    
    file_names_tmp = dir(pathcat(pathImages,'*.tif'));
    nFiles = length(file_names_tmp);
    
    T=0;
    tiffs(nFiles) = struct('file_name',[],'InfoImage',[],'stacksize',[],'tifflib',[]);
    
    %% might take so long, because tiff-information is not complete - check to create tiffs more optimally
    for i = 1:nFiles
        tiffs(i).file_name_raw = file_names_tmp(i).name;
        tiffs(i).file_name = pathcat(pathImages,file_names_tmp(i).name);
        tiffs(i).InfoImage = imfinfo(tiffs(i).file_name);
        tiffs(i).stacksize = length(tiffs(i).InfoImage);
        tiffs(i).tifflib = Tiff(tiffs(i).file_name, 'r');
        
        tiffs(i).save_name = pathcat(pathMedian,tiffs(i).file_name_raw);
        
        T = T + tiffs(i).stacksize;
    end
    width = tiffs(1).InfoImage(1).Width;
    height = tiffs(1).InfoImage(1).Height;
    
    if tiffs(1).InfoImage(1).BitDepth == 8
      bitDepth = 'uint8';
    elseif tiffs(1).InfoImage(1).BitDepth == 16
      bitDepth = 'uint16';
    else
      bitDepth = 'double';
    end
%      width
%      height
%      bitDepth
    
    if ~exist(pathMedian,'dir')
        mkdir(pathMedian);
    end
    
    options = struct();
    options.color     = false;
    options.compress  = 'no';
    options.message   = false;
    options.append    = false;
    options.overwrite = true;
    options.big       = false;
    
    [prg_str1 prg_str2] = prepare_progress_report('images processed: ',T);
    
    t_total = 0;
    tic
    for i = 1:nFiles    % load one stack of image files at once and process it with the median filter
        
        t = tiffs(i).stacksize;
        if i>1 && i<nFiles
            im = zeros(height,width,t+2*parameter.filtersize(3),bitDepth);
            im_median = zeros(height,width,0,bitDepth);
        else
            im = zeros(height,width,t+parameter.filtersize(3),bitDepth);
        end
        
        if i > 1
            offset_start = parameter.filtersize(3);
            for j=1:parameter.filtersize(3)
                tiffs(i-1).tifflib.setDirectory(j);
                im(:,:,j) = tiffs(i-1).tifflib.read;
            end
        else
            offset_start = 0;
        end
        
        for j = 1:t
            tiffs(i).tifflib.setDirectory(j);
            im(:,:,offset_start+j) = tiffs(i).tifflib.read;
        end
        
        if i < nFiles
            offset_end = parameter.filtersize(3);
            for j=1:parameter.filtersize(3)
                tiffs(i+1).tifflib.setDirectory(j);
                im(:,:,offset_start+t+j) = tiffs(i+1).tifflib.read;
            end
        else
            offset_end = 0;
        end
        
        im_median = medfilt3(im,parameter.filtersize*2+1);
        
        t_total = t_total + t;
        saveastiff(im_median(:,:,1+offset_start:offset_start+t),tiffs(i).save_name,options);
        
        now_time = toc;
        fprintf(1,prg_str1,t_total)
        fprintf(1,prg_str2,now_time)
    end
    
    for i = 1:nFiles
        tiffs(i).tifflib.close;
    end
    
    disp('\n')
end