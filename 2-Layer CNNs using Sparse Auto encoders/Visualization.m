%% Visualization
audio_or_video='audio';
student_id='20150923';
load(['Result_' student_id '_' audio_or_video sae_config '.mat']);
%load(['20150923_task1_sae.mat']);
% Draw cost curve
a=sae_config;
h1 = figure;
subplot(311); plot(cost(1:epoch,1), 'r'); title('Reconstruction cost','FontSize',18);
subplot(312); plot(cost(1:epoch,2), 'b'); title('Sparsity cost','FontSize',18);
subplot(313); plot(mean_hidden(1:epoch), 'k'); hold on; plot(1:epoch, repmat(sparsity_target, [1, epoch]) ,'k--'); hold off;
title('Mean hidden','FontSize',18);  legend('mean hidden','sparsity target');

% Draw filters
h2 = figure; draw_filters(AE.layers{1}.w , height, width);

