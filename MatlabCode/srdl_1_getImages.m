clearvars

path_Lab = 'X:\Lab';
path_Zhen = fullfile(path_Lab, 'Zhen');
path_Chun = fullfile(path_Lab, 'VR_SR_Pancrease_cine_Chun');

% fd_sr_contoured_100 = 'sr_contoured_100'; % png images with contour
% fd_Data_to_Jaehee = 'Data_to_Jaehee';
fd_sr_mat_100 = 'sr_mat_100'; % lr/sr gray/contoured slice stacks
path_mat = fullfile(path_Chun, fd_sr_mat_100);


junk = dir(fullfile(path_mat, '*.mat'));

nMatFile = length(junk);
for n = 1:nMatFile
    [~, matFileName{n}, ~] = fileparts(junk(n).name);
end
