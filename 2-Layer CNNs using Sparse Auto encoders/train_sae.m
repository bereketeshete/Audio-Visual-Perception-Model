function [weight, bias]=train_sae(data_audio,sae_config)

student_id='20150923';
audio_or_video='audio';
height = 26; width = 5; % audio patch
% Get data size descriptor
data=data_audio;
nData = size(data,2); % No of samples in the data
nFeat = size(data,1); % No of features

%% Part2: Feature standardization  [DO NOT MODIFY]
disp('Part2: Feature standardization');

data_mean = mean(data,2); 
data_std = std(data,1,2); %  std of data normalize by N
data = (data - repmat(data_mean,[1 nData]))./repmat(data_std, [1 nData]);

% now each dimension of data have zero mean, and unit variance

%% Part3: Hyperparameter setting [DO NOT MODIFY except 'MODIFY HERE']
disp('Part3: Hyperparameter setting');
nIn = nFeat;  nOut=nIn; nHidden = 100;  % Autoencoder size specification
lRate = 0.1; % learning rate

%[MODIFY HERE]
use_sparsity = true; % option to control use of sparsity term in training autoencoder
AEepoch = 5; % the number of epoch to train Autoencdoer. Modify this only if you think training 50000epoch is not enough.
sparsity_target = 0.3;  % Sparsity target: target activation for average hidden neuron values
sparsity_coeff = 1;    % Sparsity coefficients: how much do you want to weigh sparsity learning compared to reconstruction

% Variable for monitoring learning
cost = zeros(AEepoch,2);  % 1st column: reconstruction cost, 2nd column: sparsity cost
mean_hidden = zeros(AEepoch,1);

%% Part4: Initialization of Autoencoer  [DO NOT MODIFY except 'Fill in Here']
disp('Part4: Initialization of Autoencoder');

AE.nLayer = 2;
AE.layers = cell(AE.nLayer,1);
AE.memory_dim = [nIn nHidden nOut];

AE = Init_AE(AE, nData);  % Fill in Here
AE.layers{2}.w = AE.layers{1}.w';  % tied weight

%% Part5: Training Autoencoder [DO NOT MODIFY except 'Fill in Here']
disp('Part5: Training Autoencoder');

% Phase1: Feedforward
% Phase2: Error Backpropagation
% Phase3: Update weights (towards minimize cost)
for epoch = 1:AEepoch
    AE.activation{1} = data;
    %% Feed-forward
 
    for layerIdx=1:AE.nLayer
        AE = Feedforward(AE.activation{layerIdx}, AE, layerIdx);  % Fill in feedForward.m                
    end
    
    avg_act_hidden = mean(AE.activation{2},2); % Average activation of hidden neurons over all data
    mean_hidden(epoch) = mean(avg_act_hidden);
    error_signal = (AE.activation{1} - AE.activation{3});  % delta
    cost(epoch,1) = sum(sum(error_signal.^2))/nData;  % Reconstruction cost
    
    if(use_sparsity)        
        cost(epoch,2) = sum(sum((sparsity_target.* log(sparsity_target./avg_act_hidden) + (1 - sparsity_target).* log((1-sparsity_target) ./ (1-avg_act_hidden)))))/nData; % Sparsity cost
    end
    
    % Monitor learning progress
    if(use_sparsity)
        disp(['epoch = ' num2str(epoch) ', Reconstruction = ' num2str(cost(epoch,1)) ', Sparsity = ' num2str(cost(epoch,2))]);
    else
        disp(['epoch = ' num2str(epoch) ', Reconstruction = ' num2str(cost(epoch,1))]);
    end
    
    %% Backpropagation
    % Backpropagation in 2nd layer
    AE.layers{2}.err = error_signal; % Here is hint how error_signal looks like
    AE.layers{2}.grad_w = -AE.layers{2}.err * AE.activation{2}';
    AE.layers{2}.grad_b = -sum(AE.layers{2}.err,2);
    % Backpropagation in 1st layer
    if(use_sparsity)
        rho = sum(AE.activation{2},2)/nData;
        sparsity_err = repmat(-sparsity_target./avg_act_hidden + (1-sparsity_target)./(1-avg_act_hidden), 1, nData); % FILL IN HERE. sparsity error
        AE.layers{1}.err = (AE.layers{2}.w' * AE.layers{2}.err -sparsity_coeff * sparsity_err) .* AE.activation{2}.*(1-AE.activation{2} );    % FILL IN HERE. error for 1st layer weight: (weight * delta  + sparsity_coeff * sparsity_error) * derivative of sigmoid function
    else
        AE.layers{1}.err = (AE.layers{2}.w' * AE.layers{2}.err) .* AE.activation{2}.*(1-AE.activation{2}); % FILL IN HERE. error for 1st layer weight: weight * delta  * derivative of sigmoid function
    end

    AE.layers{1}.grad_w = -AE.layers{1}.err * AE.activation{1}'; % Fill in here (gradient for 1st layer weight)
    AE.layers{1}.grad_b = -sum(AE.layers{1}.err,2); % Fill in here (gradiet for 1st layer bias)
    
    %% Update [DO NOT MODIFY]
    AE = Update(AE, nData, lRate);
end
weight=AE.layers{1}.w;
bias=AE.layers{1}.b;
save(['Result_' student_id '_' audio_or_video sae_config],'AE','cost','mean_hidden');  % Save your training result [DO NOT MODIFY]

Visualization % draw cost graphs and weights for the autoendcoder
end