
function [sessionList, nSessions] = getSessions(pathMouse)
    
    str_pos = length(pathMouse)+2;
    sessions = dir(pathMouse);
    nSessions = 0;
    sessionList = {};
    for s=1:numel(sessions)
        name_tmp = pathcat(pathMouse,sessions(s).name);
        isf=strfind(name_tmp,'Session');
        if (isdir(name_tmp)==1) & (isf==str_pos)
            nSessions = nSessions+1;
            sessionList{nSessions} = name_tmp;
        end
    end
end
