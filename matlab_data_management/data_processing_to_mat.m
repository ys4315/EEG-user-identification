%figure;hold on;
fs = 160; % in Hz

parfor r=1:14 %PAY ATTENTION HERE!
    disp(r)
    for s=1:109
        if r < 10
            if s < 10
                file_name = strcat('S00',num2str(s),'R0',num2str(r),'.edf');
            else
                if s < 100
                    file_name = strcat('S0',num2str(s),'R0',num2str(r),'.edf');
                else
                    file_name = strcat('S',num2str(s),'R0',num2str(r),'.edf');
                end
            end
        else
            if s < 10
                file_name = strcat('S00',num2str(s),'R',num2str(r),'.edf');
            else
                if s < 100
                    file_name = strcat('S0',num2str(s),'R',num2str(r),'.edf');
                else
                    file_name = strcat('S',num2str(s),'R',num2str(r),'.edf');
                end
            end
        end
        [hdr, record] = edfread(file_name);
        
        rmd = mod(length(record),fs);
        ssize = (length(record) - rmd)/fs;
        % s=subject number r=trial number
        
        temp_label = zeros(109,1);
        temp_label(s) = 1;
        
        for index=2:(ssize-2)
            filename = strcat('eeg_dataset/R',num2str(r),'/s',num2str(s),'r',...
                num2str(r),'i',num2str(index-1),'.mat');
            temp_data = record(1:64, fs*(index-1)+1:fs*index);
            parsave_eeg(filename,temp_data,temp_label,r);
        end
    end
end
