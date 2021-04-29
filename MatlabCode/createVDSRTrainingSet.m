function createVDSRTrainingSet(imds, scaleFactor, upsampledDirName, residualDirName)

if ~isfolder(residualDirName)
    mkdir(residualDirName);
end
    
if ~isfolder(upsampledDirName)
    mkdir(upsampledDirName);
end

while hasdata(imds)
    % Use only the luminance component for training
    [I, info] = read(imds);
    [~, fileName, ~] = fileparts(info.Filename);
    
    I = im2double(I);
    
    upsampledImage = imresize(imresize(I,1/scaleFactor,'bicubic'),[size(I,1) size(I,2)],'bicubic');
    
    residualImage = I-upsampledImage;
    
    save([residualDirName filesep fileName '.mat'],'residualImage');
    save([upsampledDirName filesep fileName '.mat'],'upsampledImage');
    
end