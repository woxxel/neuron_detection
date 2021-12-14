
function [file] = create_tiff_stacks(path,pathStacks,nTiff)

  %%% path          - path to file directory
  %%% pathStacks    - path to created stacks
  %%% nTiff         - number of Tiffs per stack
  
% ---------------------------------------------------------------------------------------------------

  %  path
  %  pathStacks
%    if exist(pathStacks,'dir')
%      rmdir(pathStacks,'s')
%    end
  if ~exist(pathStacks,'dir')
    mkdir(pathStacks)
  end

  tiffs = struct;
  fileNames = dir(pathcat(path,'*.tif'));
  file = pathcat(path,fileNames(1).name);

  tiffs.InfoImage = imfinfo(file);
  width = tiffs.InfoImage.Width;
  height = tiffs.InfoImage.Height;

  if length(fileNames) == 1   %% only one tiff stack present -> burst
    
    img = loadtiff(file);
    if length(size(img))==2
      img = imread_big(file,8989);
    end
    nframes = size(img,3);
    nStacks = ceil(nframes/nTiff);
    
    [~, stackName, ~] = fileparts(file);
    
    for n = 1:nStacks
      idx_start = (n-1)*nTiff+1;
      idx_end = min(nframes,n*nTiff);
      
      svFile = pathcat(pathStacks,sprintf('%s_%02d.tif',stackName,n));
      disp(sprintf('saving tiff-stack #%d to %s',n,svFile))
      saveastiff(img(:,:,idx_start:idx_end),svFile);
      
    end
    
  else                        %% 
    
    if length(tiffs.InfoImage) > 1    %% check if already in stacks
      disp('tiffs are already in stacks - still want to proceed?')
    else
      
      nframes = length(fileNames);
      [~, stackName, ~] = fileparts(file);
      stackName = stackName(1:end-5);
      img = zeros(height,width,nTiff,'uint16');
      
      n=0;
      c=1;
      for i = 1:nframes
          
          file = fileNames(i).name;
          img(:,:,c) = loadtiff(pathcat(path,file));
          
          if mod(c,nTiff)==0 || i==length(fileNames)
            n=n+1;
            svFile = pathcat(pathStacks,sprintf('%s_%02d.tif',stackName,n));
            disp(sprintf('tiff-stack #%d saved to %s',n,svFile))
            saveastiff(img(:,:,1:c),svFile);
            c=0;
          end
          c=c+1;
          
      end
    end
  end
end
