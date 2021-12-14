

function tiff2h5(pathDir,pathOut)
    
    if exist(pathOut)==2
        disp(sprintf('%s already exists - removing!',pathOut))
        delete(pathOut)
    end
    
    tic
    disp(sprintf('creating h5-file @ %s',pathOut))
    
    file_names_tmp = dir(pathcat(pathDir,'*.tif'));
    
    nFiles = length(file_names_tmp);
    
    tiffs(nFiles) = struct();
    nt=0;
    for i = 1:nFiles
	tiffs(i).file_name = pathcat(pathDir,file_names_tmp(i).name);
	tiffs(i).InfoImage = imfinfo(tiffs(i).file_name);
	tiffs(i).stacksize = length(tiffs(i).InfoImage);
	tiffs(i).tifflib = Tiff(tiffs(i).file_name, 'r');
	
	nt = nt + tiffs(i).stacksize;
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
    
    dims = [tiffs(1).InfoImage(1).Height tiffs(1).InfoImage(1).Width nt];
    
    %%% initiate h5 file
    h5create(pathOut, '/DATA', dims, 'Datatype',bitDepth,'ChunkSize',[dims(1) dims(2) 1]);
    
    t = 0;
    for i = 1:nFiles
	for j = 1:tiffs(i).stacksize
	    t = t+1;
	    tiffs(i).tifflib.setDirectory(j);
	    h5write(pathOut,'/DATA',tiffs(i).tifflib.read,[1 1 t],[dims(1) dims(2) 1]);
	end
	tiffs(i).tifflib.close;
    end
    toc
end