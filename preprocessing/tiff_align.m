
classdef tiff_align < handle
    %{
    Class for aligning frames in Tiff files.
    
    DESCRIPTION:
    
    Two methods are used in sequence:
    - x-y rigid shifts to remove overall movement
    - Lucas Kanade method for non-rigid frame warping
    
    Steps:
    1) A collection of Tiff files are opened
    2) First the frames are aligned by x-y shifts to remove drift and large
    movements.
    3) Then a moving average of 2*mean_samples is used as template to correct
    each frame by warping.
    4) The corrected frames are exported as new Tiff files.
    
    The core of the methods is inspired by
    - https://scanbox.wordpress.com/2014/03/20/recursive-image-alignment-and-statistics/
    - https://xcorr.net/2014/08/02/non-rigid-deformation-for-calcium-imaging-frame-alignment/
    See Dario Ringach's Lab
    
    HOW TO USE:
    Loading files:
        - To open as a GUI, run the code without input arguments.
                tiff_align()
        A window appears where multiple Tiff files can be selected. Multiple 
        files correspond to data of a single recording that was splitted
        when saved.
        - The window appears again asking to add new files for an extra
        channel. Only the first channel is used for calculating the alignment,
        but it will be applied to all channels and exported.
        - To start the GUI from a specific folder, start as 
                tiff_align('/path/to/folder/')
        - To make the file structure programmatically, start as 
                tiff_align(file_structure)
        where file_structure is a cell array with the names of the Tiff
        files. Rows are files that are to be stitched together (equal to
        selecting multiple files with the GUI) and columns are different
        channels. Eg:
            file_structure = {'ch1_file1.tif','/ch2_file1.tif';...
                              'ch1_file2.tif','/ch2_file2.tif'}
    Parameters:
        (check class properties after LINE 75 to change them!)
        - n_basis = 16: Defines the number of splines.
        !!! THE NUMBER OF LINES IN THE FRAME SHOULD BE DIVISIBLE BY THIS !!!
        - mean_samples = 100; +/- frame range for running average for the
        template.
        - max_shift_all = []: Maximum shifts to detect in x-y direction.
        Give a number of pixels if the signal in the frames is very noisy.
        - max_shift_block = 10: Maximum initial block shifts for warping 
    
    Output:
        The code generates a new Tiff file for each given as input with the
        prefix export_
    
    MAIN FUNCTIONS:
        process_tiff: Main loop where the Tiffs are aligned and exported
        
        read_stack: Extracts a frame from the Tiff files
        
        calculate_xy_shift: Calculates rigid x-y shift for single frame alignment
        LK_align: Calculates frame warping based on template
    
        export_prepare/close/add: Handles the creation of a new Tiff file
        
    Written by: chepe@nld.ds.mpg.de
    Date: 03Feb2017
    %}
    %%
    properties (SetAccess = private)
        
        % File names and tiff handles
        tiff_info
        pathOut
        
        % Data description
        width
        height
        num_channels
        num_stacks
        stack_to_file
        
        % X-Y shifts
        xy_shift = [];
        max_shift_all = 10;
        
        % Lucas-Kanade
        n_basis = 16;
        mean_samples = 5;
        
        max_iter = 25;      
        dcorr_goal = .0005;
        damping = 1;
        max_shift_block = 10;
        
        % Exporting
        tagstruct
        trfun
        tfile
        
        % paramaters that are calculated only once
        Bf = [];
        mask = [];
        xi = [];
        yi = [];
    end
    
    %%
    methods
        
        %% Constructor, parser and destructor
        
        function obj = tiff_align(pathIn,pathOut)
            
%           % To see how the class is initialized, see make_file_structure
            % function below:
            
            % ----------- Parse input variable, which is either:
            % a single string or nothing = path for the GUI to open
            % a cell = structure of .tif file names for each channel
            disp(sprintf('image_alignment / dewarping @ %s',pathOut))
            if nargin==0
                pathIn = [];
            end
            file_names = obj.make_file_structure(pathIn);
            obj.pathOut = pathOut;
            mkdir(pathOut);
            
            % ------------ Check data
            % Do all files have the same frame size?
            tmp=imfinfo(file_names{1});
            width = tmp(1).Width;
            height = tmp(1).Height;
            for ii=2:numel(file_names)
                tmp=imfinfo(file_names{ii});
                if tmp(1).Width~=width || tmp(1).Height~=height
                    error('Not all files have the same x-y dimensions')
                end
            end
            
            % Do all channels have the same number of stacks?
            for ii=1:size(file_names,1)
                tmp=imfinfo(file_names{ii,1});
                for jj=ii+1:size(file_names,2)
                    tmp2=imfinfo(file_names{ii,jj});
                    if length(length(tmp))~=length(length(tmp2))
                        error('Not all files have the same number of stacks')
                    end
                end
            end
            
            % ------------  Open files and extract info
            
            % open files to read and create tiff_info structure.
            obj.tiff_info = struct();
            for ii=1:size(file_names,1)
                for jj=1:size(file_names,2)
                    tmp=imfinfo(file_names{ii,jj});
                    obj.tiff_info(ii,jj).tifflib = Tiff(tmp(1).Filename,'r');
                    obj.tiff_info(ii,jj).Filename = tmp(1).Filename;
                    if length(tmp)>1
                        obj.tiff_info(ii,jj).Stacks=length(tmp);
                    else
                        disp('Warning: Only 1 stack found in the file. If file size >4GB try splitting it into multiple files.')
                        obj.tiff_info(ii,jj).Stacks=1;
                    end
                end
            end
            
            % Get the total number of stacks and a mapping "stack -> corresponding file"
            obj.num_stacks = 0;
            obj.stack_to_file = [];
            for ii=1:size(obj.tiff_info,1)
                % [file_number stack_ini stack_end]
                obj.stack_to_file = [obj.stack_to_file; ii,obj.num_stacks+[1,obj.tiff_info(ii,1).Stacks]];
                obj.num_stacks = obj.num_stacks + obj.tiff_info(ii,1).Stacks;
            end
            
            tmp=imfinfo(file_names{1});
            obj.width = tmp(1).Width;
            obj.height = tmp(1).Height;
            obj.num_channels = size(file_names,2);
            obj.xy_shift = zeros(obj.num_stacks,2);
            
            disp('-- Tiff files opened...')
            
            % -------------- Do processing
            obj.process_tiff
        end
        
        function file_names = make_file_structure(obj,pathIn)   % should be changed to static method
            % Construct list of file names from input
            
            if isempty(pathIn)
                % use GUI to choose files and start from current directory
                file_names = [];
                ini_path = pwd;
            elseif iscell(pathIn)
                % file_names cell is given in input
                file_names = pathIn;
            elseif ischar(pathIn)
                % input is either path for a single file or for image folder (or for GUI)
                if strcmp(pathIn(end-3:end),'.tif')
                    file_names = {pathIn};
                elseif isdir(pathIn)
                    file_names_tmp = dir(pathcat(pathIn,'*.tif'));
                    T = length(file_names_tmp);
                    file_names = cell(T,1);
                    for t = 1:T
                        file_names{t} = pathcat(pathIn,file_names_tmp(t).name);
                    end
                else
                    file_names = [];
                    ini_path = pathIn;
                end
            else
                error('The input to tiff_reader is either a string of the path for the GUI or a cell of files {rows = file_num, columns = channel}.')
            end
            % select the file_names using a GUI, one channel at the time,
            % where for each channel multiple files can be combined
            if isempty(file_names)
                
                while true
                    ini_path
                    % get file names for next channel
                    [FileName,PathName] = uigetfile({[ini_path,'/*.tif']},'MultiSelect','on','Select .TIF files');
                    ini_path = PathName;
                    
                    % stop if canceled
                    if isnumeric(FileName)
                        break
                    end
                    
                    % convert input into cell "file_names"
                    if iscell(FileName) %i.e multiple files
                        file_names_channel = cell(length(FileName),1);
                        for ii=1:length(FileName)
                            file_names_channel{ii} = [PathName,FileName{ii}];
                        end
                    else %i.e only one file
                        file_names_channel = cell(1,1);
                        file_names_channel{1,1} = [PathName,FileName];
                    end
                    
                    % if it is the first channel, just put in list
                    if isempty(file_names)
                        file_names = file_names_channel;
                    else
                        % check that the number of files matches
                        if size(file_names,1)~= size(file_names_channel,1)
                            error('Different number of files defined as for previous channels')
                        end
                        file_names = [file_names,file_names_channel];
                    end
                end
            end
        end
        
        function delete(obj)
            % close all the tif files
            for fileNum = 1:numel(obj.tiff_info)
                obj.tiff_info(fileNum).tifflib.close;
            end
        end
        
        %% Processing loop
                
        function process_tiff(obj)
            %{
             This function does the main job of doing a first run of x-y
             alignment and then a rolling Lucas Kanade alignment.
            %}
            
            % 0) Check that the number of basis is suitable
            assert(mod(obj.height,obj.n_basis)==0,'Y dimension is not a multiple of n_basis./n Change n_basis in class properties.')
            
            % 1) Do x-y shift alignment first
            disp('-- Calculating x-y shifts...')
            tic
	    obj.calculate_xy_shift;
            disp('   Done.')
            toc
            
            % 2) Prepare output files
            obj.export_prepare;
            disp('-- Empty Tiff files generated...')
            
            % 3) Make starting average
            rolling_template = zeros(obj.height,obj.width);
            frame_count = zeros(obj.height,obj.width);
            
            for frame_ii=1:obj.mean_samples
                S = obj.read_stack(frame_ii,1);
                valS = ~isnan(S);
                rolling_template(valS) = rolling_template(valS) + S(valS);
                frame_count(valS) = frame_count(valS) + 1;
            end
            
            % 4) Go through the frames, update template, do Lucas Kanade
            % align, export
            disp('-- Calculating Lucas-Kanade warping and exporting frame by frame...')
            tic
            inform_stack = ceil(linspace(1,obj.num_stacks,11));
                        
            for frame_ii=1:obj.num_stacks
            
                % inform how we are doing
                [~,ind] = intersect(inform_stack,frame_ii);
                if ~isempty(ind)
                    time_tmp = toc;
                    disp(['     ',num2str((ind-1)*10),'% / ',num2str(frame_ii),' frames processed, time passed: ',num2str(time_tmp),' seconds'])
                end
                
                % move mean
                template_range = frame_ii + obj.mean_samples*[-1 1];
                
                % remove first image in range
                if template_range(1)>0
                    S = obj.read_stack(template_range(1),1);
                    valS = ~isnan(S);
                    rolling_template(valS) = rolling_template(valS) - S(valS);
                    frame_count(valS) = frame_count(valS) - 1;
                end
                % add new image to range
                if template_range(2)<=obj.num_stacks
                    S = obj.read_stack(template_range(2),1);
                    valS = ~isnan(S);
                    rolling_template(valS) = rolling_template(valS) + S(valS);
                    frame_count(valS) = frame_count(valS) + 1;
                end
                
                % do alignment and get weights
                [Im,dpx,dpy] = obj.LK_align(obj.read_stack(frame_ii,1),rolling_template./frame_count);
                
                % export current image and use weights for the rest
                obj.export_add(Im,frame_ii,1)
                for ch = 2:obj.num_channels
                    
                    % read image
                    img_I = obj.read_stack(frame_ii,ch);
                    fill_values = quantile(img_I(~isnan(img_I)),.01);
                    
                    % warp
                    Dx = repmat((obj.Bf*dpx),1,obj.width);
                    Dy = repmat((obj.Bf*dpy),1,obj.width);
                    Im = interp2(img_I,obj.xi+Dx,obj.yi+Dy,'linear');
                    Im(isnan(Im)) = fill_values;
                    
                    % export
                    obj.export_add(Im,frame_ii,ch)
                end
            end
            disp('   Done.')
            
            % 5) Close files
            obj.export_close
            disp('-- Finished!')
            obj.delete
            toc
        end
        
        %% Frame reader
        
        function output_image = read_stack(obj,stack_global,channel_to_extract)
            %{
             Read a stack from the files.
             Input:
             stack_number = from the total stacks (corresponding stack in file is calculated)
             channel_to_extract = from which channel the stack is obtained
             NOTE: empty pixels due to alignment are filled with NaN
            %}
            
            % if not given use object defined channel to extract
            if nargin<3
                channel_to_extract = 1;
            end
            
            % remove warnings when reading tiff
            warning('off','all')
            
            % Check the corresponding file and stack number
            file_number = find(stack_global>=obj.stack_to_file(:,2) & stack_global<=obj.stack_to_file(:,3));
            if isempty(file_number)
                disp('Required stack is not available. Returning empty stack.')
                return
            end
            stack_number = stack_global - (obj.stack_to_file(file_number,2)-1);
            
            % Go to stack and read
            obj.tiff_info(file_number,channel_to_extract).tifflib.setDirectory(stack_number);
            img = double(obj.tiff_info(file_number,channel_to_extract).tifflib.read);
            % if multi-dimensional, use only first
            img = img(:,:,1);
            
            % Shift image for alignment
            if sum(abs(obj.xy_shift(stack_global,:)))==0
                output_image = img;
            else
                output_image = nan(size(img));
                output_image(...
                    (1+max(0,obj.xy_shift(stack_global,1))):(size(img,1)+min(0,obj.xy_shift(stack_global,1))) ,...
                    (1+max(0,obj.xy_shift(stack_global,2))):(size(img,2)+min(0,obj.xy_shift(stack_global,2))) ) ...
                    = img( (1-min(0,obj.xy_shift(stack_global,1))):(size(img,1)-max(0,obj.xy_shift(stack_global,1))) ,...
                    (1-min(0,obj.xy_shift(stack_global,2))):(size(img,2)-max(0,obj.xy_shift(stack_global,2))) );
            end
        end
        
        %% Alignment functions
        
        function calculate_xy_shift(obj)
            %{
             Calculates rigid x-y shift for each frame by maximizing
             correlation between pairs of images recursively.
             obj.max_shift_all: determines the search radius for the maximum
            %}
            % restart alignment
            obj.xy_shift = zeros(obj.num_stacks,2);
            % calculate alignment
            r = xy_align(obj,1:obj.num_stacks,obj.max_shift_all);
            
            % set alignment properties
            obj.xy_shift = r.T;
        end
        
        function [Id,dpx,dpy] = LK_align(obj,img_I,img_T)
            %{
              Lucas-Kanade algorithm for non-rigid frame deformation
              Based on:
              https://xcorr.net/2014/08/02/non-rigid-deformation-for-calcium-imaging-frame-alignment/
            %}
            
            % The first time create basis functions and search mask
            if isempty(obj.Bf)
                obj.prepare_LK
            end
            
            % normalize template only once for correlation (1D)
            T_norm = img_T(:)-mean(img_T(:));
            T_norm = T_norm/sqrt(sum(T_norm.^2));
            fill_values = quantile(img_T(~isnan(img_T)),.01);
            img_T(isnan(img_T)) = fill_values;
            
            % remove NaN from input
            fill_values = quantile(img_I(~isnan(img_I)),.01);
            img_I(isnan(img_I)) = fill_values;
            
            
            %% Blockwise correlation method to initiate dpx and dpy
            
            blocksize = obj.height/obj.n_basis;
            
            dpx = zeros(obj.n_basis,1);
            dpy = zeros(obj.n_basis,1);
            for ii = 1:obj.n_basis
                % cut a chunk
                block = (ii-1)*blocksize + (1:blocksize);
                
                % get correlation matrix
                T_ = bsxfun(@minus,img_T(block,:),nanmean(img_T(block,:),1));  %% blocks: img_T - mean(img_T,1)
                I_ = bsxfun(@minus,img_I(block,:),nanmean(img_I(block,:),1));  %% blocks: img_I - mean(img_I,1)
                C = obj.mask.*fftshift(ifft2(fft2(I_).*conj(fft2(T_))));
                
                % find best shift
                [ind_y,ind_x] = find(C == max(C(:)));
		
                if any(size(ind_x)==0)
%                    C
%                    img_T
%                    T_
%                    I_
        %  		  obj.mask
                  [ind_x ind_y]
        %  		  sleep(10)
                  
        %  		  dpx(ii,1) == 0;
        %  		  dpy(ii,1) == 0;
                else
                  dpx(ii,1) = median(ind_x - floor(obj.width/2+1));
                  dpy(ii,1) = median(ind_y - floor(blocksize/2+1));
                end
            end
            
            % interpolate to get the knots
            dpx = [dpx(1);(dpx(1:end-1)+dpx(2:end))/2;dpx(end)];
            dpy = [dpy(1);(dpy(1:end-1)+dpy(2:end))/2;dpy(end)];
            
            %% Lukas Kanade method
            lambda = 0.0001*median(img_T(:))^2;
            theI = eye(obj.n_basis+1)*lambda;
            allBs = [obj.Bf.^2,obj.Bf(:,1:end-1).*obj.Bf(:,2:end)];
            
            % Initialize with given image
            Id = img_I;
            Id_norm = Id(:) - mean(Id(:));
            Id_norm = Id_norm / sqrt(sum(Id_norm.^2));
            corr_old = Id_norm'*T_norm;
            
            % Optimize weights recursively
            for ii = 1:obj.max_iter
                
                % ==== Get new weights for basis functions
                
                %gradient of template
                dTx = (Id(:,[1,3:end,end])-Id(:,[1,1:end-2,end]))/2;
                dTy = (Id([1,3:end,end],:)-Id([1,1:end-2,end],:))/2;
                
                del = img_T(:) - Id(:);
                
                %special trick for g (easy)
                gx = obj.Bf'*sum(reshape(del,size(dTx)).*dTx,2);
                gy = obj.Bf'*sum(reshape(del,size(dTy)).*dTy,2);
                
                %special trick for H (harder)
                Hx = constructH(allBs'*sum(dTx.^2,2),obj.n_basis+1)+theI;
                Hy = constructH(allBs'*sum(dTy.^2,2),obj.n_basis+1)+theI;
                
                % solve system
                dpx_ = Hx\gx;
                dpy_ = Hy\gy;
                
                % update weights
                dpx = dpx + obj.damping*dpx_;
                dpy = dpy + obj.damping*dpy_;
                
                % ==== Make corrected image and check correlation increase
                
                % Displaced template
                Dx = repmat((obj.Bf*dpx),1,obj.width);
                Dy = repmat((obj.Bf*dpy),1,obj.width);
                
                Id = interp2(img_I,obj.xi+Dx,obj.yi+Dy,'linear');
                Id(isnan(Id)) = fill_values;
                
                % check increase in correlation
                Id_norm = Id(:) - mean(Id(:));
                Id_norm = Id_norm / sqrt(sum(Id_norm.^2));
                corr_new = Id_norm'*T_norm;
                if corr_new - corr_old < obj.dcorr_goal
                    break;
                end
                corr_old = corr_new;
            end
            
        end
        
        function prepare_LK(obj)
            %{
               Define some parameters and variables only once to speed up.
            %}
            
            % Casis functions: linear B-splines (tent-shaped)
            obj.Bf = zeros(obj.height,obj.n_basis+1);
            
            knots = linspace(0,obj.height,obj.n_basis+1);
            knots = [knots(1)-(knots(2)-knots(1)),knots,knots(end)+(knots(end)-knots(end-1))];
            
            delta = 1e-10;
            x = 1:obj.height;
            for ii = 1:obj.n_basis+1
                obj.Bf(:,ii) = ...
                    (x-knots(ii))/(delta + knots(ii+1)-knots(ii)) .* (x<knots(ii+1) & x>= knots(ii)) + ...  % up
                    (knots(ii+2)-x)/(delta + knots(ii+2)-knots(ii+1)) .* (x<knots(ii+2) & x>= knots(ii+1)); % down
            end
            
            % Mask to search max correlation inside radius
            blocksize = obj.height/obj.n_basis;
            
            [mx,my] = meshgrid(1:obj.width,1:blocksize);
            center_xy = [floor(obj.width/2+1) floor(blocksize/2+1)];
            obj.mask = (mx-center_xy(1)).^2+(my-center_xy(2)).^2 < obj.max_shift_block^2;
            
            % Coordinate system
            [obj.xi,obj.yi] = meshgrid(1:obj.width,1:obj.height);
        end
        
        %% Exporting functions
        
        function export_prepare(obj)
            
            % make general tag
            obj.tagstruct = struct;
            obj.tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
            obj.tagstruct.ImageLength = obj.height;
            obj.tagstruct.ImageWidth = obj.width;
            obj.tagstruct.RowsPerStrip = obj.tiff_info(1,1).tifflib.getTag('RowsPerStrip');
            obj.tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
            obj.tagstruct.SamplesPerPixel = 1;
            obj.tagstruct.Compression = Tiff.Compression.None;
            
            data_class = class(obj.tiff_info(1,1).tifflib.read);
            switch data_class
                case {'uint8', 'uint16', 'uint32'}
                    obj.tagstruct.SampleFormat = Tiff.SampleFormat.UInt;
                case {'int8', 'int16', 'int32'}
                    obj.tagstruct.SampleFormat = Tiff.SampleFormat.Int;
                case {'single', 'double', 'uint64', 'int64'}
                    obj.tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
                otherwise
                    error('Error in determining SampleFormat');
            end
            switch data_class
                case {'uint8', 'int8'}
                    obj.tagstruct.BitsPerSample = 8;
                case {'uint16', 'int16'}
                    obj.tagstruct.BitsPerSample = 16;
                case {'uint32', 'int32'}
                    obj.tagstruct.BitsPerSample = 32;
                case {'single'}
                    obj.tagstruct.BitsPerSample = 32;
                case {'double', 'uint64', 'int64'}
                    obj.tagstruct.BitsPerSample = 64;
                otherwise
                    error('Error in determining BitsPerSample');
            end
            % transform to integers function
            eval(['obj.trfun = @',data_class,';'])
            
            % open file handles:
            obj.tfile = cell(size(obj.tiff_info,1),size(obj.tiff_info,2));
            for file_num = 1:size(obj.tiff_info,1)
                for channel_num = 1:size(obj.tiff_info,2)
                    % delete previous
                    [pathstr, fname, fext] = fileparts(obj.tiff_info(file_num,channel_num).Filename);
%                      mkdir(obj.pathOut);
                    file_name = pathcat(obj.pathOut,[fname,fext]);
                    if exist(file_name,'file')
                        delete(file_name)
                    end
                    % make new file
                    obj.tfile{file_num,channel_num} = Tiff(file_name, 'w');
                end
            end
            
        end
        
        function export_add(obj,Im,stack_num,channel_num)
            
            % convert
            Im = obj.trfun(Im);
            
            % add
            file_num = find(stack_num>=obj.stack_to_file(:,2) & stack_num<=obj.stack_to_file(:,3));
            if isempty(file_num)
                disp('Required stack is not available.')
                return
            end
            stack = stack_num - (obj.stack_to_file(file_num,2)-1);
            stacks_to_use = obj.stack_to_file(file_num,2):obj.stack_to_file(file_num,3);
            
            % save
            obj.tfile{file_num,channel_num}.setTag(obj.tagstruct);
            obj.tfile{file_num,channel_num}.write(Im);
            % expand
            if stack ~= stacks_to_use(end)
                obj.tfile{file_num,channel_num}.writeDirectory();
            end
        end
        
        function export_close(obj)
            % close files
            for file_num = 1:size(obj.tiff_info,1)
                for channel_num = 1:size(obj.tiff_info,2)
                    obj.tfile{file_num,channel_num}.close();
                end
            end
            
        end
        

    end
    
end

%% HELPER FUNCTIONS TO CALCULATE X-Y SHIFT

function r = xy_align(obj,stacks_to_do,max_shift)
    %{
    Calculates rigid x-y shift for each frame by maximizing correlation
    between pairs of images recursively.
    r = running data structure with:
        .I mean image
        .n number of frames in mean
        .T x-y shifts for the frames
    max_shift = search radius to maximize correlation
    Based on:
    https://scanbox.wordpress.com/2014/03/20/recursive-image-alignment-and-statistics/
    %}
    
    
    if length(stacks_to_do)==1
        % if only one stack, assign initial values to parameters
        r.I= obj.read_stack(stacks_to_do);
        r.n = 1;
        r.T = [0 0];
    else
        % split into two groups and run again recursively
        r_input = xy_align(obj,stacks_to_do(1:floor(end/2)),max_shift);
        r_goal = xy_align(obj,stacks_to_do(floor(end/2)+1:end),max_shift);
        
        % use the average of the selected channel to get alignment
        [shift_y,shift_x] = max_corr(r_input.I,r_goal.I,max_shift);
        
        % shift input image (fill empty spaces with NaN)
        if abs(shift_y)+abs(shift_x)>0
            tmp = NaN*zeros(size(r_input.I));
            tmp( (1+max(0,shift_y)):(size(r_input.I,1)+min(0,shift_y)) , (1+max(0,shift_x)):(size(r_input.I,2)+min(0,shift_x)) ) = ...
                r_input.I( (1-min(0,shift_y)):(size(r_input.I,1)-max(0,shift_y)) , (1-min(0,shift_x)):(size(r_input.I,2)-max(0,shift_x)) );
            r_input.I = tmp;
        end
        
        % combine images, number of stacks and transforms
        r.I = (r_input.n*r_input.I + r_goal.n*r_goal.I)/(r_input.n+r_goal.n);
        r.n = r_input.n+r_goal.n;
        r.T = [ones(size(r_input.T,1),1)*[shift_y shift_x] + r_input.T ; r_goal.T];
    end
    
end

function [shift_y,shift_x] = max_corr(img_input,img_goal,max_shift)
    %{
    Finds the x-y shift inside a max_shift radius which maximizes correlation
    between the images.
    max_shift = set to NaN if no search radius is given
    %}
    
    % cut edges with NaN (from previous transforms) using the center of the
    % images as reference
    nan_y = isnan(img_input(:,ceil(size(img_input,2)/2)).*img_goal(:,ceil(size(img_input,2)/2)));
    nan_x = isnan(img_input(ceil(size(img_input,1)/2),:).*img_goal(ceil(size(img_input,1)/2),:));
    img_input = img_input(find(~nan_y),find(~nan_x));
    img_goal = img_goal(find(~nan_y),find(~nan_x));

    % cut both images to be squared using the minimum size of img_input as reference
    %%% that makes it square around the center of cropped image - is that wanted?
    N = floor(min(size(img_input))/2);
    yidx = floor(size(img_input,1)/2) -N + 1 : floor(size(img_input,1)/2) + N;
    xidx = floor(size(img_input,2)/2) -N + 1 : floor(size(img_input,2)/2) + N;
    img_input = img_input(yidx,xidx);
    img_goal = img_goal(yidx,xidx);

    % Calculate cross correlation and find position of maximum
    C = fftshift(real(ifft2(fft2(img_input).*fft2(rot90(img_goal,2)))));    %% shouldn't this also be conj? (conj=rot90, I guess, but conj might be faster)
    if ~isnan(max_shift)
        % restrict search shifts
        [xi,yi] = meshgrid((1:2*N)-N+1);
        mask = (xi.^2+yi.^2) < (max_shift^2);
        C = C.*mask;
    end
    [ind_y,ind_x] = find(C == max(C(:)));
    
    % Use the position of the maximum relative to center to determine shift
    shift_y = N - ind_y;
    shift_x = N - ind_x;
end

%%  HELPER FUNCTION FOR LUCAS KANADE ALGORITHM

function H2 = constructH(Hd,ns)
    H2d1 = Hd(1:ns)';
    H2d2 = [Hd(ns+1:end);0]';
    H2d3 = [0;Hd(ns+1:end)]';
    
    H2 = spdiags([H2d2;H2d1;H2d3]',-1:1,ns,ns);
end


