function cnnRep2(imageFolder,outFolder)
addpath([ 'Matconvnet/matlab/']);
vl_setupnn;

load([outFolder 'net-epoch-3.mat']);
net=vl_simplenn_tidy(net);
net=cnn_imagenet_deploy(net);
%pathname = strcat(outFolder,'benchmarkingFeatures_epoch20.mat');
%s = (str2double(TaskIDStr));
%str = sprintf('Benchmark_features_%d.mat',s);
%pathname = strcat([outFolder,str]);
%disp('yoo');
for k = 1:1
    data = [];
    labels = [];
    fname = 'imdb.mat';
   % fname = sprintf('data_batch_%d.mat',k);
    str = strcat(imageFolder,fname);
    arr = load(str);
    data = arr.images.data;
    labels = arr.images.labels;
%disp('heyy');    

%fprintf('slurm%s step %s\n',TaskIDStr,StepIDStr);
TaskIDStr = getenv('SLURM_ARRAY_TASK_ID');
StepIDStr = getenv('CUSTOM_ARRAY_STEP_ID');
if ~isempty(TaskIDStr) && ~isempty(StepIDStr)
    startIdx = (str2double(TaskIDStr)-1) * str2double(StepIDStr) + 1;
    endIdx = startIdx+str2double(StepIDStr)-1;
    s = (str2double(TaskIDStr));
    str = sprintf('cvl_features_%d.mat',s);
else
    disp('No task ID specified');
    startIdx = 1;
    endIdx = 400000;
    str  = 'Train_features_1.mat';
  % s = (str2double(TaskIDStr));
   str = sprintf('Train_features_%d.mat',str);
end

   
%s = (str2double(TaskIDStr));
%str = sprintf('Train_features_%d.mat',s);
pathname = strcat([outFolder,str]);
%disp('hi');
c =0;label = [];
for i = startIdx:endIdx
%disp('hiii');
    c = c+1;
    disp(sprintf('%d-%d',k,i));
    im_ar = data(:,:,:,i);
    im_ar = reshape(im_ar,48,128,1);
   % imshow(im_ar)
    im_ar = single(im_ar);
    net.layers{end}.type = 'softmax';
    res = vl_simplenn(net, im_ar) ;
    l = c;
    scores1(l,:) = [transpose(squeeze(gather(res(16).x)))];
    label(l)= transpose(labels(i));
    %[M,idx] = max( scores1(l,:));
    %pred_labels(l,:) = idx;
    %act_labels(l,:) = labels(i);
end
end
save(pathname,'scores1','label','-v7.3');
