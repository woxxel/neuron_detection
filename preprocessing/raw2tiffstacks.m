
function raw2tiffstacks(pathRaw,nTiff)
  
  tic
%    pathConf = '/home/wollex/Data/Documents/Uni/2016-XXXX_PhD/Japan/Work/Data/512x512x8989.conf';
  pathConf = '/home/mizuta/AlexCode/512x512x8989.conf';
%    pathRaw = dir(pathcat(pathIn,'*.raw'));
%    pathRaw = pathcat(pathIn,pathRaw(1).name);
  
  [pathFolder,fileName,ext] = fileparts(pathRaw);
  disp(sprintf('in folder %s converting raw file %s to tiff stack',pathFolder,fileName))
%    pathRaw = pathcat(pathFolder,[fileName,'.raw']);
  pathH5 = pathcat(pathFolder,[fileName,'.h5']);
  
  if ~exist(pathH5,'file')
    system(sprintf('h5import %s -c %s -outfile %s',pathRaw,pathConf,pathH5));
  end
  
  h52tiffstacks(pathH5,nTiff)
  delete(pathH5)
  toc
end


function h52tiffstacks(pathH5,nTiff)

  %%% path          - path to file directory
  %%% pathStacks    - path to created stacks
  %%% nTiff         - number of Tiffs per stack
  
  % ---------------------------------------------------------------------------------------------------

  %  path
  %  pathStacks
  %  if exist(pathStacks,'dir')
  %    rmdir(pathStacks,'s')
  %  end

%    tiffs = struct;
%    fileNames = dir(pathcat(pathFolder,'*.h5'));

%    file = pathcat(pathFolder,fileNames(1).name);
  [pathFolder,fileName,~] = fileparts(pathH5);
  pathTif = pathcat(pathFolder,[fileName,'.tif']);
  
  disp(sprintf('writing data from %s to %s',pathH5,pathTif))

  I = h5read(pathH5,'/DATA');

  height = size(I,1);
  width = size(I,2);
  t = size(I,3);
  nStacks = ceil(t/nTiff);

  %  img = zeros(height,width,nTiff,'uint16');

  %  c=1;
  for i = 1:nStacks
    
    idx_first = (i-1)*nTiff+1;
    idx_last = min(t,i*nTiff);
    
    img = I(:,:,idx_first:idx_last);
  %        tiffld = Tiff(pathcat(path,file),'r');
  %      img(:,:,c) = tiffld.read;
      
  %      if mod(c,nTiff)==0 || i==length(fileNames)
%      svFile = sprintf('%s/stack%02d.tif',pathFolder,i);
    disp(sprintf('tiff#=%d, save stuff to %s',i,pathTif))
    saveastiff(img,pathTif);
  %        c=0;
  %      end
  %      c=c+1;
  %    end
  end
end