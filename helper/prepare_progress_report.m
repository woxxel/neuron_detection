

function [prg_str1, prg_str2] = prepare_progress_report(dspl_string,num)%%% preparing progress output
    req_blanks1 = ceil(log10(num))+1;
    time_string = ', elapsed time (seconds): ';
    length_time = 6;
    req_blanks2 = length(time_string) + length_time;
    
    req_blanks = req_blanks1 + req_blanks2;
    fprintf(1,[dspl_string '(' num2str(num) ') ' blanks(req_blanks)]);
    backspace_string = '';
    for j = 1:req_blanks
        backspace_string = strcat(backspace_string,'\b');
    end
    prg_str1 = [backspace_string '%-' num2str(req_blanks1) 'd'];
    prg_str2 = [time_string '%' num2str(length_time) '.1f'];
end