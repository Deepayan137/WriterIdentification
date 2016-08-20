%{
X = load('cifar_features.mat');
%dataEncode = X.scores1(1:5,:);
data = X.scores1;
 numClusters = 30 ;
[means, covariances, priors] = vl_gmm(data, numClusters);
numDataToBeEncoded = 100;
enc = vl_fisher(data, means, covariances, priors);

numFeatures = 5000 ;
dimension = 1 ;
data = rand(dimension,numFeatures) ;

numClusters = 100 ;
[means, covariances, priors] = vl_gmm(data, numClusters);
numDataToBeEncoded = 1000;
Encoded = rand(dimen:qsion,numDataToBeEncoded);
enc= vl_fisher(Encoded, means, covariances, priors);

%}
function cifar_gmm(imageFolder,outFolder)
run vlfeat-0.9.20/toolbox/vl_setup;
%rc = load('rescov.mat');
%r = load('res.mat');:
%load('arrayA.mat');
name = 'fisherVector.mat';
pathname = strcat([outFolder,name]);
X = load([outFolder,'PCAexperimental_features.mat']);
%load('records.mat');
load([outFolder,'PCAbenchmark_features.mat']);

TaskIDStr = getenv('SLURM_ARRAY_TASK_ID');
StepIDStr = getenv('CUSTOM_ARRAY_STEP_ID');
fprintf('slurm%s step %s\n',TaskIDStr,StepIDStr);
if ~isempty(TaskIDStr) && ~isempty(StepIDStr)
    startIdx = (str2double(TaskIDStr)-1) * str2double(StepIDStr) + 1;
    endIdx = startIdx+str2double(StepIDStr)-1;
   % str = sprintf('data_batch_%d.mat',(str2double(TaskIDStr)));
else
    disp('No task ID specified');
    startIdx =1;
    endIdx = 1000;
   % str = 'SYnthdata_batch_1.mat';
end
%{
y = [];Y=[];
for j = 1:11
    disp(j);

    
    fname = sprintf('hw1ktest_features_%d.mat',j);
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
save('HW/PCAhw1ktest_features.mat','Y','-v7.3');
disp('saved');

keyboard;
%}
numClusters =5;
%[COEFF,XSCORE] = princomp(X.scores1);
%[COEFF,ySCORE] = princomp(y.scores1);
data = transpose(X.Y);
[means, covariances, priors,ll,posteriors] = vl_gmm(data, numClusters);
%save('benchmark_mat/gmmModel.mat','means','covariances','priors','ll','posteriors');
%load('gmmModel.mat');
%{
pri = priors;
pos = posteriors;
posSum = sum(pos,2);
alpha = zeros(100,1);
 for k = 1:100
     alpha(k) = posSum(k)/(posSum(k)+28);
 end
 cov = covariances;
sumpro  =sum((pos*X.Y),2);
sumcov = sum((pos*(X.Y.*X.Y)),2);
M = zeros(100,64);
C = zeros(100,64);
c = (cov.*cov+means.*means)-(means.*means);
 for i = 1:100
     M(i,:) =  alpha(i)*(sumpro(i)/posSum(i))+ (1 -alpha(i))*means(:,i)';
     C(i,:)  = alpha(i)*(sumcov(i)/posSum(i))+ (1-alpha(i))*c(:,i)';
     P(i,:) = (alpha(i)/size(data))*posSum(i)+(1-alpha(i))*pri(:,i)';
 end
 for n =1:100
 mini = min(M(n,:));
 max = 0.0052;
 res(n,:) =  -1 +2.*(M(n,:) - mini)/ (max - mini);
 rescov(n,:) =  -1 +2.*(C(n,:) - mini)/ (max - mini);
 end
res = single(M');
rescov = single(C');
%}
%begin  = 1;
%over = 0;
for i = startIdx:endIdx
disp(sprintf('%d',i))
%:qd = dir('words/');
%n = length(d);
   
    % disp(sprintf('%d',i));
    % start = ai;
begin = (i-1)*2000+1;
over = begin +2000 -1;

%over = over+array(i);
dataEncode = transpose(Y(begin:over,:));
%begin = over+1;
%keyboard;
%{
  if i >1
    startIdx = endIdx+1;
     endIdx = startIdx+a(i)-1;
    
    dataEncode = transpose(Y(startIdx:endIdx,:));
     else
        startIdx = 1;
     endIdx = startIdx+a(i)-1;
     
    dataEncode = transpose(Y(startIdx:endIdx,:));
    end
    %data = single(rand(64,5000));
     
   
    %numDataToBeEncoded = 100;
%}  
    enc(:,i)= vl_fisher(dataEncode,means,covariances, priors);
    end

save(pathname,'enc','-v7.3');
%}
