clear all
close all
clc


%% Spectrum Loading
SAVENAME = 'MouseLeg4_FROCT';

SaveImagepath = '\Image\';
warning off
mkdir([SaveImagepath SAVENAME]);
warning on

SaveMatpath = '\Mat\';
warning off
mkdir([SaveMatpath SAVENAME]);
warning on

AngleNum = 1;
% only 1 angle frame was privide in this Data
% If you would like to obtain the full 60 Angle raw data please contact the author
Image_DiffAngle_Volume = zeros(2048,2048,AngleNum);
% Image_DiffAngle_MultiFrame_Volume = zeros(2048,2048,8,AngleNum);

for i = 1:AngleNum
%     i = 30
%     1:AngleNum
    fp=fopen(['Data\RawSepctrum.bin'],'r');
    A1=fread(fp,'uint16');
    fclose(fp);
    ALINE_NUM=2048;
    PIXEL_NUM=2048;
    FRAME_NUM=max(size(A1))/ALINE_NUM/PIXEL_NUM;
%     FRAME_NUM = 11;
%% Reference Subtraction

    Bmode_Image_Intensity=zeros(FRAME_NUM,ALINE_NUM,PIXEL_NUM);
    Bmode_Image_Volume=zeros(ALINE_NUM,PIXEL_NUM,FRAME_NUM);

    for frame_index=1:1:FRAME_NUM
        BmodeFrame=zeros(ALINE_NUM,PIXEL_NUM);
        Reference_Spectrum=zeros(1,PIXEL_NUM);
        BmodeFrame=reshape(A1((frame_index-1)*PIXEL_NUM*ALINE_NUM+1:frame_index*PIXEL_NUM*ALINE_NUM),PIXEL_NUM,ALINE_NUM)';

        for n=1:1:PIXEL_NUM
            Reference_Spectrum(n)=median(BmodeFrame(:,n));
        end
    
        for n=1:1:ALINE_NUM
            for m=1:1:PIXEL_NUM
                BmodeFrame(n,m)=BmodeFrame(n,m)-Reference_Spectrum(m);
            end
        end

%% Image Processing
        Bmode_Image=zeros(ALINE_NUM,PIXEL_NUM);
        Bmode_Image_F = zeros(ALINE_NUM,PIXEL_NUM);
        Bmode_Spectrum=zeros(ALINE_NUM,PIXEL_NUM);
        Aline_Spectrum=zeros(1,PIXEL_NUM);
        Aline_Spectrum_Linear=zeros(1,PIXEL_NUM);

%% 
        for n=1:1:ALINE_NUM
            Aline_Spectrum=BmodeFrame(n,:);
            Aline_Spectrum_Linear=interp1(1:PIXEL_NUM,Aline_Spectrum,1:PIXEL_NUM,'spline');
            Bmode_Spectrum(n,:)=Aline_Spectrum_Linear;
        end
        
%% 
        ImgThreshold=zeros(ALINE_NUM,PIXEL_NUM);
        Filter=zeros(1,ALINE_NUM);
        CenterFrequency=2048-345;
        FrequencyBandWidth=185;
        HorizontalSpectrumTemp=zeros(1,ALINE_NUM);
        for m=1:1:ALINE_NUM
            Filter(m)=exp(-(m-CenterFrequency)*(m-CenterFrequency)/FrequencyBandWidth/FrequencyBandWidth);
        end
    
        for n=1:1:PIXEL_NUM
            HorizontalSpectrumTemp= Bmode_Spectrum(:,n);
            Bmode_Spectrum(:,n)=ifft(fft(HorizontalSpectrumTemp).*Filter');
        end
        
        for n=1:1:ALINE_NUM
            Aline_Spectrum_Linear=Bmode_Spectrum(n,:);
            Aline_Postfft=fftshift(fft(Aline_Spectrum_Linear));
            Bmode_Image(n,:)=Aline_Postfft;
%           Bmode_Image_Intensity(frame_index,n,:)=abs(Aline_Postfft);
        end

        ImgThreshold=log(abs(Bmode_Image') + 1);
        m = max(max(ImgThreshold));
        ImgThreshold = ImgThreshold/m;
        Bmode_Image_Volume(:,:,frame_index)=ImgThreshold;

    end


	
    
    % sum-2
    Bmode_Image_sum = squeeze(sum(Bmode_Image_Volume, 3));
    Bmode_Image_ave = Bmode_Image_sum/FRAME_NUM;
    figure(2),imagesc(Bmode_Image_ave);colormap(gray);
    
    
    
%% SaveData
    imwrite(Bmode_Image_ave, [SaveImagepath SAVENAME '\',num2str(i-1),'.png' ]);
    
    Image_DiffAngle_Volume(:,:,i) = Bmode_Image_ave;

end


ImageDiffAngleMultiFrameVolume = Image_DiffAngle_Volume;
MatSave_Dir2 = [SaveMatpath SAVENAME,'\','ImageDiffAngleMultiFrameVolume.mat'];
save(MatSave_Dir2, 'ImageDiffAngleMultiFrameVolume')




