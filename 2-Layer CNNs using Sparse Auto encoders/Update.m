function AE = Update(AE, nData, lRate)

AE.layers{2}.w = AE.layers{2}.w - lRate*AE.layers{2}.grad_w/nData;
AE.layers{2}.b = AE.layers{2}.b - lRate*AE.layers{2}.grad_b/nData;

AE.layers{1}.w = AE.layers{1}.w - lRate*AE.layers{1}.grad_w/nData;
AE.layers{1}.b = AE.layers{1}.b - lRate*AE.layers{1}.grad_b/nData;

tied_weight = (AE.layers{2}.w' + AE.layers{1}.w)/2;

AE.layers{1}.w = tied_weight;  % to statisfy tied weight constraint
AE.layers{2}.w = tied_weight'; % to statisfy tied weight constraint
end