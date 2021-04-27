clearvars

exts = {'.jpg','.png'};
fileNames = {'sherlock.jpg','car2.jpg','fabric.png','greens.jpg','hands1.jpg','kobi.png', ...
    'lighthouse.png','micromarket.jpg','office_4.jpg','onion.png','pears.png','yellowlily.jpg', ...
    'indiancorn.jpg','flamingos.jpg','sevilla.jpg','llama.jpg','parkavenue.jpg', ...
    'peacock.jpg','car1.jpg','strawberries.jpg','wagon.jpg'};
filePath = [fullfile(matlabroot,'toolbox','images','imdata') filesep];
filePathNames = strcat(filePath,fileNames);
testImages = imageDatastore(filePathNames,'FileExtensions',exts);

figure(1), clf
montage(testImages)

%% original image
indx = 1; % Index of image to read from the test image datastore
Ireference = readimage(testImages,indx);
Ireference = im2double(Ireference);
figure(2), clf
imshow(Ireference)
title('High-Resolution Reference Image')

%% upsample
scaleFactor = 2;
Ilowres = imresize(Ireference,scaleFactor,'bicubic');
figure(3), clf
imshow(Ilowres)
title('Low-Resolution Image')

%% VDSR
Iycbcr = rgb2ycbcr(Ireference);
Iy = Iycbcr(:,:,1);
Icb = Iycbcr(:,:,2);
Icr = Iycbcr(:,:,3);

[nrows,ncols,np] = size(Ilowres);

% Upscale the luminance and tw
Iy_bicubic = imresize(Iy,[nrows ncols],'bicubic');
Icb_bicubic = imresize(Icb,[nrows ncols],'bicubic');
Icr_bicubic = imresize(Icr,[nrows ncols],'bicubic');

% Pass the upscaled luminance component, Iy_bicubic, through the trained VDSR network. Observe the activations (Deep Learning Toolbox) from the final layer (a regression layer). The output of the network is the desired residual image.
load('trainedVDSR-Epoch-100-ScaleFactors-234.mat');
Iresidual = activations(net, Iy_bicubic, 41);
Iresidual = double(Iresidual);
figure(4), clf
imshow(Iresidual,[])
title('Residual Image from VDSR')

Isr = Iy_bicubic + Iresidual;
% Concatenate the high-resolution VDSR luminance component with the upscaled color components. Convert the image to the RGB color space by using the ycbcr2rgb function. The result is the final high-resolution color image using VDSR.

Ivdsr = ycbcr2rgb(cat(3,Isr,Icb_bicubic,Icr_bicubic));
figure(5), clf
imshow(Ivdsr)
title('High-Resolution Image Obtained Using VDSR')
