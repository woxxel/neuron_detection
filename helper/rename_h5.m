

function rename_h5(pathMouse)
  
  [sessionList, nSessions] = getSessions(pathMouse);
  
  for s = 1:nSessions
    
    pathSession = sessionList{s};
%      pathSession
    
%      fileNames = dir(pathcat(pathSession,'*.tif'));
    fileNames = dir(pathcat(pathSession,'thy*'));
    fileTiff = pathcat(pathSession,fileNames(1).name);
    [~,fname,~] = fileparts(fileTiff);
%      fname
%      fileTiff
    fileH5_in = pathcat(pathSession,'ImagingData_MF1_LK1.h5');
    fileH5_out = pathcat(pathSession,sprintf('%s.h5',fname));
%      fileH5_in
%      fileH5_out
    if exist(fileH5_in)
      sprintf('rename %s to %s',fileH5_in,fileH5_out)
      delete(fileTiff)
      movefile(fileH5_in,fileH5_out)
    end
  end
  
  
end