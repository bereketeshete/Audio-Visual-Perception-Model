function x = Feedforward(prev_x, AE, layerIdx)

    nData = size(prev_x,2); 
    x_hat =  AE.layers{layerIdx}.w * prev_x + repmat(AE.layers{layerIdx}.b,[1 nData]); % FILL IN HERE (x_hat = weighted sum of prev_x)
    % Note: activation function for first layer: sigmoid, second layer:linear
    if(layerIdx == 1)
       x = 1./(1+exp(-x_hat));  % FILL IN HERE (sigmoid activation)   
    else
       x = x_hat; % FILL IN HERE (linear activation)     
    end
    AE.activation{layerIdx+1} = x;
    
    x=AE;
end




