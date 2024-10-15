clear;
close all;
clc;

%% add path
% pathname = 'E:\DL-SC-FROCRT\Code\';
% addpath(genpath(pathname))

%% DataLoad
% Non-registration
datapathname = '\Mat\';

filename = 'ImageDiffAngleMultiFrameVolume.mat';% mat from Conjugate_Free.m 
% only 1 angle frame was privide in this Data
% If you would like to obtain the full 60 Angle raw data please contact the author
filename_nomat=filename(1:end-4);

ImageDataLoad = load([datapathname filename]);
ImageVolume = ImageDataLoad.ImageDiffAngleMultiFrameVolume;
clear ImageDataLoad;


%% Preprocessing

savename = ['E:\DL-SC-FROCRT\Image\20240401\MouseLeg3_FROCT_OCT\'];
warning off
mkdir([savename]);
warning on

ImageBmode_FORShow = ImageVolume(:,:,1);
figure(1),imagesc(ImageBmode_FORShow),colormap(gray)

% Rotation Center
SIZE_ImageVolume = size(ImageVolume);

ALINE_NUM = SIZE_ImageVolume(1);
PIXEL_NUM = SIZE_ImageVolume(2);
ANGLE_NUM = SIZE_ImageVolume(3);
PIXEL_RESIZE = 1024;
ALINE_RESIZE = 512;
SCALE = 0.65;

% Random threshold 
Mask_i_EachBmode = zeros(1, ANGLE_NUM);
Mask_j_EachBmode = zeros(1, ANGLE_NUM);
for i = 1:ANGLE_NUM
    Threshold_k = 0.55;
    n = 1025;
    m = 785;
    
    Picture_Load = ImageVolume(:,:,i);
    Picture_Double = im2double(Picture_Load);
%     Picture_Scale = imresize(Picture_Double, [2048, 2048*SCALE]);
    Picture_Scale = Picture_Double;
    Picture_Scale_Shape = size(Picture_Scale);
    
    Picture_Bin = imbinarize(Picture_Scale, Threshold_k); 
    figure(101),imagesc(Picture_Bin);colormap(gray);
    
    Mask = zeros(Picture_Scale_Shape(1), Picture_Scale_Shape(2));
    Mask(n-20 :n+20, m-40:m+40) = 1;
    Picture_Masked = Picture_Bin .* Mask;
% part1
    [Mask_i, Mask_j] = find(Picture_Masked == 1);
    Mask_i_EachBmode(1, i) = round(sum(Mask_i) / length(Mask_i));
    Mask_j_EachBmode(1, i) = round(sum(Mask_j) / length(Mask_j));
end

% Image Segmentation
CropImage_Volume = zeros(PIXEL_RESIZE, ALINE_RESIZE, ANGLE_NUM);
Mask_i_AveValue = round(mean(Mask_i_EachBmode));
Mask_j_AveValue = round(mean(Mask_j_EachBmode));
% ALINE_i = Mask_i_AveValue-Mask_i_EachBmode;
% ALINE_j = Mask_j_AveValue-Mask_i_EachBmode;
for i = 1:ANGLE_NUM
    
    Pict_Load = ImageVolume(:,:,i);
    % Remove Middle Highlight-Line
    Threshold_H = 3;
    Pic_Part1 = Pict_Load(1:1024-Threshold_H, :);
    Pic_Part2 = Pict_Load(1025+Threshold_H:end,:);
    Pic_Part = [Pic_Part1; Pic_Part2];
    
    % Shift-cycle
    ALINE_i = Mask_i_AveValue-Mask_i_EachBmode(i);
    ALINE_j = Mask_j_AveValue-Mask_j_EachBmode(i);
    PictShift = circshift(Pic_Part,[ALINE_i,ALINE_j]);
    
    % Image-corp
    ALINE_j_MIN = min(Mask_j_AveValue)-1;
    Distortion_compensation = 95;
    PictCrop = PictShift(Mask_i_AveValue:(Mask_i_AveValue+(1024+512-Threshold_H-Mask_i_AveValue)*2)-Distortion_compensation,...
                         1:Mask_j_AveValue*2);
%     figure(103),imagesc(PictCrop);colormap(gray);
    
    % Image-Scale
    PictScaled =  imresize(PictCrop,[PIXEL_RESIZE,ALINE_RESIZE]);
    
    % Image-SaveImage
    
    PictScaled_max = max(max(PictScaled));
    PictSave = PictScaled / PictScaled_max;
    figure(103),imagesc(PictSave);colormap(gray);
    imwrite(PictSave, [savename '\Center\',num2str(i-1),'.png']);
    
    CropImage_Volume(:, :, i) = double(PictSave);
    
    
end
SaveMatpath = ['E:\DL-SC-FROCRT\Mat\20240401\MouseLeg3_FROCT_OCT\'];
SaveMatName = 'MouseLeg3_FROCT_OCT_Raw';
MatSave_Dir2 = [SaveMatpath SaveMatName,'.mat'];
save(MatSave_Dir2, 'CropImage_Volume')


%% CycleCal
ImageVolume = CropImage_Volume;

fidelity=28.50; % lamda  1500  100  75 50 118.5
Lamdax=1.25;
Lamday=1.0;
Lamdat=1; % lamda_x y t 
parall=12; %  lamda_L1 300  
iteration=25;
gpu = 1;%Use GPU
mu=1;

psf_parmat = 10;
psf_iter = 8;
% PSFRatio = 10.5;
% Gaussian = GaussianDistri([PSFRatio,PSFRatio], 0, 0, PSFRatio/3, PSFRatio/2, 0); %(GaussianSize, u1, u2, sigma1, sigma2, rou)
% GaussianNor = Gaussian;
% PSF = GaussianNor;


Gaussian2 = load('E:\DL-SC-FROCRT\Code\SupplementFunction\psf_Measured.mat');
GaussianNor1 = Gaussian2.PSF;
GaussianNor2 = NOR(imresize(GaussianNor1,[70,70]));
GaussianNor3 = NOR(circshift(GaussianNor2,[0 1]));
PSF = GaussianNor2;
figure(201),imagesc(PSF);

ImageVolumeShape = size(ImageVolume);
Hessian_NonHanced = zeros(ImageVolumeShape); 
DeCovBlind_Hessian = zeros(ImageVolumeShape);
RawFROCT = zeros(ImageVolumeShape);
for i = 1:ImageVolumeShape(3)
        
    f = ImageVolume(:,:,i);
    g_Hessian = SparseHessian_Hance(f,fidelity,Lamdax,Lamday,Lamdat,parall,iteration,gpu,mu);
    g_Hessian = gather(g_Hessian);
        
    g_DeCovBlind = SparseHessian_DeConv_BlindPSF_20240710(f,fidelity+1.5,Lamdax,Lamday,Lamdat,parall,iteration,gpu,mu,PSF,psf_parmat, psf_iter);
    g_DeCovBlind = gather(g_DeCovBlind);
        
    % Normalized
    g_Hessian_max = max(g_Hessian(:));
    g_Hessian_NOR = g_Hessian / g_Hessian_max;
    
    g_DeCovBlind_max = max(g_DeCovBlind(:));
    g_DeCovBlind_NOR = g_DeCovBlind / g_DeCovBlind_max;
        
    f_max = max(f(:));
    f_nomor = f/f_max;
        
    % ImageShow
    figure(13),imagesc(f_nomor);colormap(gray),title('ori');
    figure(14),imagesc(g_Hessian_NOR);colormap(gray),title('g NonPSF');
    figure(16),imagesc(g_DeCovBlind_NOR);colormap(gray),title('g Hessain Blind PSF');
        
    % Image Save
    Hessian_NonHanced(:,:,i) = g_Hessian_NOR;
    DeCovBlind_Hessian(:,:,i) = g_DeCovBlind_NOR;
    RawFROCT(:,:,i) = f_nomor;
        
    savename1 = [savename '\Hessian'];
    warning off
    mkdir([savename1]);
    warning on
    imwrite(g_Hessian_NOR, [savename1 '\',num2str(i-1),'.png']);
        
    savename4 = [savename '\BlindHessian'];
    warning off
    mkdir([savename4]);
    warning on
    imwrite(g_DeCovBlind_NOR, [savename4 '\',num2str(i-1),'.png']);
        
end
    
savematname0 = ['E:\DL-SC-FROCRT\Mat\20240401\MouseLeg3_FROCT_OCT\RawFROCT\'];
warning off
mkdir([savematname0]);
warning on

savematname1 = ['E:\DL-SC-FROCRT\Mat\20240401\MouseLeg3_FROCT_OCT\Hessian\'];
warning off
mkdir([savematname1]);
warning on

savematname2 = ['E:\DL-SC-FROCRT\Mat\20240401\MouseLeg3_FROCT_OCT\BlindHessian\'];
warning off
mkdir([savematname2]);
warning on
    
RawFROCT = RawFROCT;
HessianOnly = Hessian_NonHanced;
BlindHessian = DeCovBlind_Hessian;
SAVENAME = 'MouseLeg3_OCT';
MatSave_Dir0 = [savematname0 ,'\', SAVENAME '-RawFROCTVolume.mat'];
save(MatSave_Dir0, 'RawFROCT')
MatSave_Dir1 = [savematname1 ,'\', SAVENAME '-HessianOnlyVolume.mat'];
save(MatSave_Dir1, 'HessianOnly')
MatSave_Dir2 = [savematname2 ,'\', SAVENAME '-BlindHessianVolume.mat'];
save(MatSave_Dir2, 'BlindHessian')

