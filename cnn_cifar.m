function [net, info] = cnn_cifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

%run(fullfile(fileparts(mfilename('fullpath')), ...
 % '..', '..', 'matlab', 'vl_setupnn.m')) ;
%addpath([ 'Matconvnet/matlab/']);
%vl_setupnn;
addpath([ 'Matconvnet_cuda/matlab/']);
vl_setupnn;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', sprintf('cifar-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.batchNormalization = true;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
opts.jitter = 1;
% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'lenet'
    net = cnn_cifar_init('networkType', opts.networkType) ;
  case 'nin'
    net = cnn_cifar_init_nin('networkType', opts.networkType) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end

%net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;
;
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images =imdb.images.data(:,:,:,batch) ;
%{
ima = imdb.images.data(:,:,:,batch) ;
si =1;
imo = zeros(48,128,1,size(batch,2));
opts.jitter = 1;
for i=1:size(batch,2)
img = ima(:,:,:,i);
maxVal=max(max(img));
if(opts.jitter)
	rVal = rand;
  if(rVal<0.5)
	if(rand<0.5)
		thres=-0.5+rand*0.5;
                 tform = affine2d([1 0 0; thres 1 0; 0 0 1]);   %horizontal shear
                 img = imwarp(img,tform,'FillValues',maxVal);
         end
         if(rand>0.5)
               padVal=40;
               lPad = ceil(rand*padVal);
               rPad = ceil(rand*padVal);
               tPad = ceil(rand*padVal);
               bPad = ceil(rand*padVal);
               img = padarray(img,[tPad lPad],maxVal,'pre');
               img = padarray(img,[bPad rPad],maxVal,'post');
         end
   end
end
%img = imresize(img, [48, 128,1,
%img = reshape(img,[48,128,1,[]])
img = imresize(img, [48,128]);
img = single(img);       
%keyboard;
imo(:,:,1,si) = img;
si = si+1;
end
images =single(imo); 
%}
labels = imdb.images.labels(1,batch) ;
%if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
%if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:9, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 9), 3]);

%if any(cellfun(@(fn) ~exist(fn, 'file'), files))
%  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
%  fprintf('downloading %s\n', url) ;
 % untar(url, opts.dataDir) ;
%end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data,32,32,1,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels'; % Index from 1
  
  sets{fi} = repmat(file_set(fi), size(data{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
%data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
%{
if opts.contrastNormalization
  z = reshape(data,[],420000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 1, []) ;
end

if opts.whitenData
  z = reshape(data,[],420000) ;
  W = z(:,set == 1)*z(:,set == 1)'/420000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 1, []) ;
end
%}

%clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
%imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
%imdb.meta.classes = clNames.label_names;
