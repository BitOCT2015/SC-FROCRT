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

% The original code was written by % https://weisongzhao.github.io/Sparse-SIM/
% and modified by SC-FROCRT in an innovative way


function g=SparseHessian_DeConv_BlindPSF_20240710(f,fidelity,Lamdax,Lamday,Lamdat,paral1,iteration,gpu,mu,PSF,psf_parmat,psf_iter)

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
f_flag=size(f,3);
if f_flag<3
%     Lamdat=0;
    f(:,:,end+1:end+(3-size(f,3)))=repmat(f(:,:,end),[1,1,3-size(f,3)]);
    disp('Number of data frame is smaller than 3, the t or z-axis of continuity was turned off (conti=0)');
end
f=f./max(f(:));
f=single(f);
[sx,sy,sz]=size(f);
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
    f=gpuArray(f);
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
g_update = (fidelity/mu)*f;
for iter = 1:iteration
    tic;
    g_update = fftn(g_update);
    if iter>1
        g = real(ifftn(g_update./normlize));
    else
        g = real(ifftn(g_update./(fidelity/mu)));
    end
    g_update = (fidelity/mu)*f;
    
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
    
    % PSF Decov & Blind Decov
    
     if iter>psf_parmat && iter<iteration-3
        xshow = NOR(gather(sum(g,3)));
        [J_Blind,DePSF] = deconvblind(xshow,PSF,psf_iter-5,[]); 
        g_update(:,:,2) = NOR(J_Blind);

     end
     if iter >= (iteration-3)
         xshow = NOR(gather(sum(g,3)));
         J_Blind1 = deconvlucy(xshow,PSF,psf_iter); 
         g_update(:,:,2) = NOR(J_Blind1);
         figure(1111),imagesc(DePSF)
     end
    
    
    
    ttime = toc;
    disp(['  iter ' num2str(iter) ' | ' num2str(iteration) ', took ' num2str(ttime) ' secs']);
end
g(g<0)=0;
if f_flag<3
    g=g(:,:,2);
end
clear bxx byy bzz bxz bxy byz bl1 f normlize g_update
