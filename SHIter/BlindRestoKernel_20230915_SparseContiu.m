% function g=BlindRestoKernel_20230602_SparseContiu(Spectrum,CMatrix, fidelity,Lamdax,Lamday,Lamdat,parall,iteration,gpu,mu)
function g=BlindRestoKernel_20230602_SparseContiu(Spectrum,UpdatedU, fidelity,Lamdat,parall,iteration,gpu,mu)
disp('CMtrix Update');

if nargin < 6 || isempty(iteration)
    iteration=100;
end
if nargin < 7 || isempty(gpu)
    gpu=cudaAvailable;
end
if nargin < 8 || isempty(mu)
    mu=1;
end
% progressbar('Sparsity reconstruction');
Lamdat=single(sqrt(Lamdat));
f_flag=size(Spectrum,3);
if f_flag<3
    Lamdat=0;
    Spectrum(:,:,end+1:end+(3-size(Spectrum,3)))=repmat(Spectrum(:,:,end),[1,1,3-size(Spectrum,3)]);
    disp('CMtrix Update Use Single!');
end
% Spectrum=Spectrum./max(Spectrum(:));
Spectrum=single(Spectrum);
% [sx,sy,sz]=size(Spectrum);
% [~,sy,sz]=size(Spectrum);
% [~, sy_spectrum, ~]=size(UpdatedU);
% sizeg=[sy_spectrum,sy,sz];

% % Blind?
[sy,~,sz]=size(Spectrum);
[sy_spectrum,~, ~]=size(UpdatedU);
sizeg=[sy_spectrum,sy,sz];

xxfft=operation_xx(sizeg);
yyfft=operation_yy(sizeg);
zzfft=operation_zz(sizeg);
xyfft=operation_xy(sizeg);
xzfft=operation_xz(sizeg);
yzfft=operation_yz(sizeg);

UpdatedUMatrixfft=operation_CMatrix(UpdatedU,sizeg);

operationfft=xxfft+yyfft+(Lamdat^2)*zzfft+ xyfft+...
              2*(Lamdat)*xzfft+2*(Lamdat)*yzfft;
normlize = single((fidelity/mu)*UpdatedUMatrixfft +parall^2 +operationfft);
% normlize = single((fidelity/mu) +parall^2 +operationfft);
clear xxfft yyfft zzfft xyfft xzfft yzfft operationfft
if gpu==1
    Spectrum=gpuArray(Spectrum);
    normlize =gpuArray(normlize);
    bxx = gpuArray.zeros(sizeg,'single');
    byy = bxx;
    bzz = bxx;
    bxy = bxx;
    bxz = bxx;
    byz = bxx;
    bl1 = bxx;
else
    bxx = zeros(sizeg,'single');
    byy = bxx;
    bzz = bxx;
    bxy = bxx;
    bxz = bxx;
    byz = bxx;
    bl1 = bxx;
end
% MultiOpration_g_update = Multi_MatrixOpration(CMatrix,Spectrum,false);
MatrixOprationFlag = false;
MultiOpration_g_update = Multi_MatrixOprationBlind(UpdatedU,Spectrum,MatrixOprationFlag);
% MultiOpration_g_update = Multi_MatrixOpration(UpdatedU,Spectrum,MatrixOprationFlag);
g_update = (fidelity/mu)*MultiOpration_g_update;
% g_update = (fidelity/mu)*Spectrum;
for iter = 1:iteration
    tic;
    g_update = fftn(g_update);
    if iter>1
        g = real(ifftn(g_update./normlize));
    else
        g = real(ifftn(g_update./((fidelity/mu))));
    end
    g_update = (fidelity/mu)*MultiOpration_g_update;
    
    [Lxx,bxx]=iter_xx(g,bxx,1,gpu);
    g_update = g_update+Lxx;
    
    [Lyy,byy]=iter_yy(g,byy,1,gpu);
    g_update = g_update+Lyy;
    
    [Lzz,bzz]=iter_zz(g,bzz,Lamdat^2,gpu);
    g_update = g_update+Lzz;
    
    [Lxy,bxy]=iter_xy(g,bxy,1,gpu);
    g_update = g_update+Lxy;
    
    [Lxz,bxz]=iter_xz(g,bxz,2*Lamdat,gpu);
    g_update = g_update+Lxz;
    
    [Lyz,byz]=iter_yz(g,byz,2*Lamdat,gpu);
    g_update = g_update+Lyz;
    
    [Lsparse,bl1]=iter_sparse(g,bl1,parall,gpu);
    g_update = g_update+Lsparse;
    ttime = toc;
    disp(['  iter ' num2str(iter) ' | ' num2str(iteration) ', took ' num2str(ttime) ' secs']);
%     figure(1133),imagesc(g_update(:,:,2)),colormap(gray);
    
end
g(g<0)=0;
if f_flag<3
    g=g(:,:,2);
end
clear bxx byy bzz bxz bxy byz bl1 f normlize g_update