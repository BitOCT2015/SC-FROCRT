%***************************************************************************
% Sparse deconvolution algorithm core
%***************************************************************************
% function g = SparseHessian_core(f,fidelity,contiz,paral1,iteration,gpu,mu)
%-----------------------------------------------
%Source code for
% argmin_g { ||f-g ||_2^2 +||gxx||_1+||gxx||_1+||gyy||_1+lamdbaz*||gzz||_1+2*||gxy||_1
% +2*sqrt(lamdbaz)||gxz||_1+ 2*sqrt(lamdbaz)|||gyz||_1+2*sqrt(lamdbal1)|||g||_1}
%f           input data
%fidelity    fidelity {example:150}
%contiz      continuity along z-axial {example:1}
%paral1      sparsity {example:15}
%iteration   iteration {default:100}
%gpu         if using CUDA {default:cudaAvailable}
%mu          lagrangian multiplier{default:1}
%------------------------------------------------
%Output:
%   g
%
%***************************************************************************
% Written by WeisongZhao @ zhaoweisong950713@163.com
% Version 1.0.3
% if any bugs is found, please just email me or put an issue on the github.
%***************************************************************************
% hhttps://weisongzhao.github.io/Sparse-SIM/
% *********************************************************************************
% It is a part of publication:
% Weisong Zhao et al. Sparse deconvolution improves the resolution of live-cell
% super-resolution fluorescence microscopy, Nature Biotechnology, X, XXX-XXX (2021).
% https://doi.org/10.1038/s41587-021-01092-2
% *********************************************************************************
%    Copyright 2018~2021 Weisong Zhao et al.
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the Open Data Commons Open Database License v1.0.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%    Open Data Commons Open Database License for more details.
%
%    You should have received a copy of the 
%    Open Data Commons Open Database License
%    along with this program.  If not, see:
%    <https://opendatacommons.org/licenses/odbl/>.
%***************************************************************************

% fidelity=200; % lamda 1000
% contiz=10;  % lamda_t 
% parall=120; % lamda_L1 50
function g=BlindPSFKernel_20230823(f_Decoved,fidelity,Lamdax,Lamday,Lamdat,paral1,iteration,gpu,mu,PSF,psf_parmat,psf_iter)

if nargin < 5 || isempty(iteration)
    iteration=100;
end
if nargin < 6 || isempty(gpu)
    gpu=cudaAvailable;
end
if nargin < 7 || isempty(mu)
    mu=1;
end
% progressbar('Sparsity reconstruction');
Lamdat=single(sqrt(Lamdat));
f_flag=size(f_Decoved,3);
if f_flag<3
    Lamdat=0;
    f_Decoved(:,:,end+1:end+(3-size(f_Decoved,3)))=repmat(f_Decoved(:,:,end),[1,1,3-size(f_Decoved,3)]);
    disp('Number of data frame is smaller than 3, the t or z-axis of continuity was turned off (conti=0)');
end
f_Decoved=f_Decoved./max(f_Decoved(:));
f_Decoved=single(f_Decoved);
[sx,sy,sz]=size(f_Decoved);
sizeg=[sx,sy,sz];
xxfft=operation_xx(sizeg);
yyfft=operation_yy(sizeg);
zzfft=operation_zz(sizeg);
xyfft=operation_xy(sizeg);
xzfft=operation_xz(sizeg);
yzfft=operation_yz(sizeg);
operationfft=(Lamdax^2)*xxfft+(Lamday^2)*yyfft+(Lamdat^2)*zzfft+ 2*(Lamday*Lamdax)*xyfft+...
              2*(Lamdax*Lamdat)*xzfft+2*(Lamday*Lamdat)*yzfft;
normlize = single((fidelity/mu) +paral1^2 +operationfft);
clear xxfft yyfft zzfft xyfft xzfft yzfft operationfft
if gpu==1
    f_Decoved=gpuArray(f_Decoved);
    normlize =gpuArray(normlize);
    bxx = gpuArray.zeros(sizeg,'single');
    byy = bxx;
    bzz = bxx;
    bxy =bxx;
    bxz =bxx;
    byz = bxx;
    bl1 =bxx;
else
    bxx = zeros(sizeg,'single');
    byy = bxx;
    bzz = bxx;
    bxy = bxx;
    bxz = bxx;
    byz = bxx;
    bl1 = bxx;
end
g_update = (fidelity/mu)*f_Decoved;
for iter = 1:iteration
    tic;
    g_update = fftn(g_update);
    if iter>1
        g = real(ifftn(g_update./normlize));
    else
        g = real(ifftn(g_update./(fidelity/mu)));
    end
    g_update = (fidelity/mu)*f_Decoved;
    
    [Lxx,bxx]=iter_xx(g,bxx,Lamdax^2,gpu);
    g_update = g_update+Lxx;
    
    [Lyy,byy]=iter_yy(g,byy,Lamday^2,gpu);
    g_update = g_update+Lyy;
    
    [Lzz,bzz]=iter_zz(g,bzz,Lamdat^2,gpu);
    g_update = g_update+Lzz;
    
    [Lxy,bxy]=iter_xy(g,bxy,2*Lamday*Lamdax,gpu);
    g_update = g_update+Lxy;
    
    [Lxz,bxz]=iter_xz(g,bxz,2*Lamdat*Lamdax,gpu);
    g_update = g_update+Lxz;
    
    [Lyz,byz]=iter_yz(g,byz,2*Lamdat*Lamday,gpu);
    g_update = g_update+Lyz;
    
    [Lsparse,bl1]=iter_sparse(g,bl1,paral1,gpu);
    g_update = g_update+Lsparse;
    
    % PSF Decov
    
     if iter>psf_parmat
        xshow = gather(g_update(:,:,2));
%         figure(202),imagesc(xshow),colormap(gray);
        J5 = deconvlucy(NOR(xshow),PSF,psf_iter); 
%         figure(203),imagesc(J5),colormap(gray);
        g_update(:,:,2) = J5;

     end
    
    % Updated PSF
    g_updateUP=g_update;
    CMatrUpdate = BlindRestoKernel_20230721_SparseContiu(Spectrum,g_updateUP, fidelity-10,Lamdat,parall,2,gpu,mu);
    CMatrUpdate2 = CMatrUpdate(:,:,sz);
    
    normlize_Cmtarix = normlize_Stern+(fidelity/mu)*operation_CMatrix(CMatrUpdate2, sizeg);
    normlize_Cmtarix = gpuArray(normlize_Cmtarix);
    
    
    
    if iter>psf_parmat
        xshow = gather(g_update(:,:,2));
%         figure(202),imagesc(xshow),colormap(gray);
        J5 = deconvlucy(NOR(xshow),PSF,psf_iter); 
%         figure(203),imagesc(J5),colormap(gray);
        g_update(:,:,2) = J5;

    end 
    
    
    ttime = toc;
    disp(['  iter ' num2str(iter) ' | ' num2str(iteration) ', took ' num2str(ttime) ' secs']);
end
g(g<0)=0;
if f_flag<3
    g=g(:,:,2);
end
clear bxx byy bzz bxz bxy byz bl1 f normlize g_update
