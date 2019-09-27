function [W2grad,feature,Indices] = mslProcessGRADIENTS_Original(img,W,theta)

I = im2double(img);

% Parameter Initialisation
[m,n,~] = size(I);
epsilon = 0.1; 
count = 1; 
scale = 8;

%%
% Convert m x n x 3 image into [(8x8x3) x count] patches
i = 1;
while (i < m - (scale - 2))
    j = 1;
    while (j< n-(scale-2)) %(j < 512)
        patch_temp = I(i:i+(scale-1),j:j+(scale-1),:);
        patches(:,count) = reshape(patch_temp,[],1);
        count = count+1;
        j = j+8;
    end    
    i = i+8;
end

%% Preprcessing
% Subtract mean patch (hence zeroing the mean of the patches)
% meanEachPatch = mean(patches);
meanPatch = mean(patches,2);  
patches = bsxfun(@minus, patches, meanPatch);

% Apply ZCA whitening
sigma = patches * patches' / (count-1);
[u, s, ~] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;

%% Gradient Initializations
imageChannels = 3;     % number of channels. For RGB = 3
patchDim   = scale;        % patch dimension. Autoencoder trained with size 8x8x3
visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
hiddenSize  = size(W,1);           % number of hidden units 

sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term       

[W2grad,feature, Indices] = mslCompute_Gradients(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, patches);

end