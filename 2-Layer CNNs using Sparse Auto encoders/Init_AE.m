function AE_out = Init_AE(AE, nData)
% Initialize weight, bias, gradient, and error signal

for i=1:AE.nLayer
    AE.layers{i}.w = -0.01 + 0.02.*rand(AE.memory_dim(i+1),AE.memory_dim(i)); % FILL IN HERE (initialization of weight: uniform random numbers between -0.01 and 0.01)
    AE.layers{i}.b = zeros(AE.memory_dim(i+1),1); % FILL IN HERE (initialization of bias: zero)
    AE.layers{i}.grad_w = zeros(size(AE.layers{i}.w));    % same size with weight
    AE.layers{i}.grad_b = zeros(size(AE.layers{i}.b));    % same size with bias
    AE.layers{i}.err = zeros(AE.memory_dim(i+1),nData);   % delta in Error Backpropagation Formula
end

% Define activation in AE
AE.activation = cell(AE.nLayer+1,1);

for i=1:AE.nLayer+1
    AE.activation{i} = zeros(AE.memory_dim(i), nData);
end

AE_out = AE;
end