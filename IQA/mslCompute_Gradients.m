function [W2grad,result, Indices] = mslCompute_Gradients(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)


W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%% 
m = size(data, 2);

z_2 = bsxfun(@plus, W1 * data, b1);
a_2 = sigmoid(z_2); % 

z_3 = bsxfun(@plus, W2 * a_2, b2);
a_3 = z_3; 


diff = a_3 - data;
delta_3 = diff;  

W2grad = delta_3 * a_2'/m;% + lambda * W2; % 25 64

y = W2grad'*(data);% - repmat(b2grad,1,size(data,2)));

logit = log(y./(ones(size(y))-y));

result1 = abs(logit);
result2 = angle(logit);

weight_process = (W2grad(1:64,:))';
weight_process = weight_process - (sum(weight_process(:))/length(weight_process(:)));
weight_process = weight_process./((ones(size(weight_process,1),1))*max(abs(weight_process)));
          
kurt = kurtosis(weight_process,0,2);
Ind = find(kurt > 3);
Indices = ones(400,1);
Indices(Ind(1:length(Ind)),1) = 2;

Ind2 = find(kurt < 2);
Indices(Ind2(1:length(Ind2)),1) = 1;

result1 = result1.*(repmat(Indices,1,size(result1,2)));
result2 = result2.*(repmat(Indices,1,size(result2,2)));

result = [result1(:);result2(:)];

%%

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));    
end

end
