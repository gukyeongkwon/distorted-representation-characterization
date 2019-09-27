function result = mslProcessGRADIENTS_Distorted(img,W2grad,Indices)

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

y = W2grad'*(patches);
logit = log(y./(ones(size(y))-y));

result1 = abs(logit);
result2 = angle(logit);

result1 = result1.*(repmat(Indices,1,size(result1,2)));
result2 = result2.*(repmat(Indices,1,size(result2,2)));

result = [result1(:);result2(:)];

end