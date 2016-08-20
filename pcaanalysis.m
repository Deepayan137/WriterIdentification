function pcaanalysis(imageFolder,outFolder)
y = [];Y=[];
for j = 1:10
    disp(j);

    
    fname = sprintf('benchmark_features_%d.mat',j);
    str = strcat(imageFolder,fname);
    y = load(str);
    x = (y.scores1)';
[nsamples, nfeatures] = size(x);
sigma = x * x' ./ nfeatures;
[U S V] = svd(sigma);
xRot = U' * x;
temp =0;
lambda = sum(S,2);
var = sum(lambda);
for k = 1:size(S,1)
    temp = temp +lambda(k);
    if (temp/var)> 0.99
       % disp(k);Benchmarkdata1024
        break;
    else 
        continue;
    end
end
epsilon = 0.1;
xHat =  U(:, 1:k)' * x;
xPCAWhite = diag(1./sqrt(diag(S) + epsilon)) * U' * x; 
xZCAWhite = U * xPCAWhite;
 v = xZCAWhite/norm(xZCAWhite);
 scores1 = v';
    Y =[Y; scores1];
end    
save('icdar_feat/PCAbenchmark_features.mat','Y','-v7.3');
disp('saved');
