clearvars

%% data source
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

%% 
path_LR = 'X:\Lab\Zhen\SRDL\Images\LR_Chun';
path_SR = 'X:\Lab\Zhen\SRDL\Images\SR_Chun';

for n = 1:1
    matFile = fullfile(path_mat, [matFileName{n}, '.mat']);
    load(matFile);
    
    for m = 1:size(vol_gray_lr, 3)
        ffn_lr = fullfile(path_LR, [matFileName{n}, '_', num2str(m),  '.mat']);
        I = vol_gray_lr(:, :,  m);
        save(ffn_lr, 'I');    
    
        ffn_sr = fullfile(path_SR, [matFileName{n}, '_', num2str(m),  '.mat']);
        I = vol_gray_sr(:, :,  m);
        save(ffn_sr, 'I');    
    end
end