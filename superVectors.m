function superVectors(imageFolder,outFolder)
load([outFolder,'PCAexperimental_features.mat']);
num =70;
%load('data/gmmModel.mat');
%pos = posteriors;
try
    rng(1);
obj = gmdistribution.fit(Y(1:size(Y,1),:),num,'Regularize',0.1);
catch exception
    disp('There was an error fitting the Gaussian mixture model')
    error = exception.message;
end
disp('GMM object created');
means = (obj.mu)';
sigma = obj.Sigma;
priors = (obj.PComponents)';
%X = load('benchmark_mat/PCAbenchmark_features.mat');
X = load([outFolder,'PCAbenchmark_features.mat']);
pos = posterior(obj,X.Y(1:size(X.Y,1),:));
for m = 1:num

suma(:,m) = (priors(m)*pos(:,m))./(pos*priors);
 
end

disp('Posteriors calculated');
%{
for i = 1:70
    disp(i);
    for j = 1:200
    
     G(j,:,i)= normpdf(Y(j,:),means(:,i)',covariances(:,i)');
    end
end

sumPos = [];
for i = 1:70
    pro(:,:,i) = priors(i)*G(:,:,i);
     s = sum(pro);
end
for j = 1:70
    pos(:,j) = pro(:,:,j)/s(:,:,j);
end
pos = pos';
%pri = priors;
%cov = covariances;
%sigma = diag(cov);
%posSum = sum(pos,2);

alpha = zeros(70,10);
%}
pos = suma';

 for k = 1:num
     
     %newpos = pos(:,start:endIdx);
     for l = 1:1000
      start = (l-1)*2000+1;
      endIdx = start+2000-1;
     posSum =sum(pos(k,start:endIdx),2);
     alpha(k,l) = (posSum)/(posSum+68);
     end
 end
disp('alpha obtained');
%}disp
% M = zeros(10,64);
 product = [];
 suma = zeros(num,64);
 for i = 1:num
     
     for j = 1:1000
         start = (j-1)*2000+1;
         endIdx = start+2000-1;
         product = pos(i,start:endIdx)*X.Y(start:endIdx,:);
         newPosSum = sum(pos(i,start:endIdx),2);
         %alpha(i,j) = newPosSum/(newPosSum+68);
         M(j,:) = (alpha(i,j)/newPosSum)*product + (1-alpha(i,j))*means(:,i)';
     end
     newMeans(:,:,i) = M;
 end
disp('acquired the new Means');
%{
for a = 1: num
 for row = 1:size(Y,2)
  for col = 1:size(Y,2)
   if row ==col
     si(row,col,a) = sigma(row,col,a);
   else
     si(row,col) = 0;
   end
  end
 end
end
%}
%keyboard;
  %sigma = diag(covariances');
  for p = 1:1000
      for q = 1:num
      
      normalized(p,:,q) = sqrt(sigma(:,:,q))* newMeans(p,:,q)'*(priors(q)^(-0.5));
      end
  end
disp('Normalized');
         % save('benchmark_mat/superVector.mat','normalized','newMeans','-v7.3');
%disp('saved');
%load('benchmark_mat/superVector.mat');
sv = [];
for p = 1:1000
    sv = [];
    for q = 1:num
        sv = [sv normalized(p,:,q)];
    end
    SV(p,:) = sv;
end

%save('benchmark_mat/superVector.mat','normalized','newMeans','SV','-v7.3');
save('icdar_feat/superVector1.mat','SV','-v7.3');
disp('saved');
