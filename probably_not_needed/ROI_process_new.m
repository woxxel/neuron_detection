

function ROI_process_new(start_idx,end_idx,sz_median,do_align,makeStacks,redo,clean_up,pathMouse)
    
    %% if no path is provided, search mousefolder by GUI
    if nargin < 8
      pathMouse = uigetdir('Choose a mouse folder to process');
    end
%      pathMouse = '/media/mizuta/Analyze_AS1/linstop/231';
    [sessionList, nSessions] = getSessions(pathMouse);
    
    %% construct suffix for filenames if not specified
    suffix_MF = sprintf('_MF%d',sz_median);
    suffix_LK = sprintf('_LK%d',do_align);
    suffix = sprintf('%s%s',suffix_MF,suffix_LK);
    
    parameter = set_parameter(sz_median);
    
    for s = start_idx:end_idx
      
      pathSession = pathcat(pathMouse,sprintf('Session%02d',s));
      path = set_paths(pathSession,suffix_MF,suffix_LK,suffix)
      
      if ~exist(path.CNMF,'file') || redo
	
        disp(sprintf('\t ### Now processing session %d ###',s))
        path.handover = path.images;
        
%          if ~exist(path.H5,'file') || redo
          if makeStacks
%              pathTmp = pathcat(path.session,'images');
    	    pathTmp = path.session;
            path.tiff = create_tiff_stacks(pathTmp,path.images,parameter.nsubFiles);
            [fol,fname,~] = fileparts(path.tiff);
            path.H5 = pathcat(fol,sprintf('%s.h5',fname));
          end
          disp(path.H5)
          %% median filtering
          if sz_median
            if ~exist(path.median,'dir') || redo==5
              median_filter(path.images,path.median,parameter);
            else
              disp(sprintf('Path: %s already exists - skip median calculation',path.median))
            end
            path.handover = path.median;
          else
            disp('---- median filtering disabled ----')
          end
          
          % image alignment
          if do_align
            if ~exist(path.LKalign,'dir') || redo>=4
              tiff_align(path.handover,path.LKalign);
            else
              disp(sprintf('Path: %s already exists - skip image dewarping',path.LKalign))
            end
            path.handover = path.LKalign;
          else
            disp('---- LK-dewarping disabled ----')
          end
          if ~exist(path.H5,'file') || redo>=3
            tiff2h5(path.handover,path.H5);
          end
%          end
        
%          if ~exist(path.reduced,'file') || redo>=2
%              reduce_data(path.H5,path);
%          else
%              disp(sprintf('Path: %s already exists - skip reduced image calculation',path.reduced))
%          end
        
        
%        if ~exist(path.CNMF,'file') || redo>=1
%            disp('do CNMF')
%            CNMF_frame(path,parameter.npatches,parameter.K,parameter.tau,0);
%        else
%            disp(sprintf('Path: %s already exists - skip CNMF',path.CNMF))
%        end
      end
      
%        if ~exist(path.CNMF_post,'file')
%            ROI_post_procession_CNMF(path,parameter);
%        else
%            disp(sprintf('Path: %s already exists - skip CNMF post-procession',path.CNMF_post))
%        end
      if clean_up
        try
          rmdir(path.images,'s');
          delete(path.tiff)
        catch
        end
%          try
%            delete(path.H5)
%          catch
%          end
      end
    end
end



function [parameter] = set_parameter(sz_median)
    
    parameter = struct();
    
    %% parameter for preCellDeconv
    %% for tiff-image->stack conversion
    parameter.nsubFiles = 2000;
    
    %% median filtering
    parameter.filtersize = [sz_median,sz_median,1];
    
    %% parameter for CNMF
    parameter.npatches = 9;                      % how many patches are processed in parallel
    K = 1600;                                    % first guess of the number of neurons to be found
    parameter.K = ceil(K/parameter.npatches);                           
    parameter.tau = 8;                           % guess of average neuron radius (in pixel)
    
    %% parameter for post-procession
%      parameter.sd = 40;                 % multiple of STD for thresholding ROI images
%      parameter.thr_size = [20 400];     % upper and lower threshold for ROI size (realistic pyramidal neuron size ~20mum length, wikipedia)
%      parameter.thr_pos = [5 507];       % threshold for ROI-position (5 off the border)
%      parameter.perc = 0.2;              % threshold for fraction of common pixels between ROIs
    
    %% parameter for session matching
%      parameter.max_dist = 12;
%      parameter.num_ses_thr = 3;
%      parameter.SI_thr = 0.5;
end


function [path] = set_paths(pathIn,suffix_MF,suffix_LK,suffix)
    
    path = struct();
    
    path.session = pathIn;
    
    %% for pre-procession
    path.images = pathcat(path.session,'imageStacks');
    path.median = pathcat(path.images,sprintf('median%s',suffix_MF));
    path.LKalign = pathcat(path.images,sprintf('LKalign%s',suffix_MF));
    path.reduced = pathcat(path.session,sprintf('reduced%s.mat',suffix));
    
    %% for CNMF algorithm
%      path.H5 = pathcat(path.session,sprintf('ImagingData%s.h5',suffix));
%      path.H5 = ''
    path.CNMF = pathcat(path.session,sprintf('resultsCNMF%s.mat',suffix));
    path.plotContour = pathcat(path.session,sprintf('contour%s.png',suffix));
    path.CNMF_post = pathcat(path.session,sprintf('ROIs_CNMF%s.mat',suffix));    
    
end
