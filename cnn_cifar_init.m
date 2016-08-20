function net = cnn_cifar_init(varargin) 
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
%opts.weightInitMethod = 'xavierimproved' ;
opts.weightInitMethod = 'gaussian' ;
opts.model = 'hwnet' ;
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.preTrainedFile='data/cifar-lenet_500fonts/net-epoch-6.mat';
lr = [.1 2] ;
opts.jitter=1;
opts = vl_argparse(opts, varargin) ;
switch opts.model
case 'hwnet'
net.meta.normalization.imageSize = [32,32, 1] ;
    net = hwnet(net, opts) ;
   % bs =32;
case 'finetune-hwnet'
load(opts.preTrainedFile);
net.layers(end-1:end)=[];
net = add_block(net, opts, '10', 1, 1, 256, 100, 1, 0) ;
net.layers(end) = [] ;


if opts.batchNormalization, net.layers(end) = [] ; end

if ~opts.batchNormalization
  %lr = logspace(-2, -4, 60) ;
      lr = logspace(-2.5, -5, 40);
else
                %lr = logspace(-1, -4, 20) ;
      lr = logspace(-2.5, -5, 40);
end

net.meta.trainOpts.learningRate = lr ;

                                %net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.weightDecay = 0.0005 ;
%net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

net= vl_simplenn_tidy(net) ;
switch lower(opts.weightInitMethod)
case {'xavier', 'xavierimproved'}
   net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end

net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
                                              
%net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

otherwise
    error('Unknown model ''%s''', opts.model);

end
%{
if ~opts.batchNormalization
  %lr = logspace(-2, -4, 60) ;
    lr = logspace(-2.5, -5, 40);
    else
      %lr = logspace(-1, -4, 20) ;
        lr = logspace(-2.5, -5, 40);
        end
        
net.meta.trainOpts.learningRate = lr ;

%net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.weightDecay = 0.0005 ;
%net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

net= vl_simplenn_tidy(net) ;
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
      net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
      end
      
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
%}
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end

%{
% Define network CIFAR10-quick
net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(5,5,1,16, 'single'), zeros(1, 16, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;


% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(5,5,16,256, 'single'), zeros(1,256,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ; % Emulate caffe

% Block 3
%net.layers{end+1} = struct('type', 'conv', ...
 %                          'weights', {{0.05*randn(8,8,256,256, 'single'), zeros(1,256,'single')}}, ...
 %                          'learningRate', lr, ...
 %                          'stride', 1, ...
 %                          'pad', 2) ;
%net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'pool', ...
 %                          'method', 'avg', ...
 %                          'pool', [2 2], ...
 %                          'stride', 2, ...
 %                          'pad', [0 1 0 1]) ; % Emulate caffe

% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(8,8,256,64, 'single'), zeros(1,64,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,64,100, 'single'), zeros(1,100,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;
%}
function net = hwnet(net, opts)
net.layers = {} ;

net = add_block(net, opts, '1', 5, 5, 1, 64, 1, 2) ;
net = add_norm(net, opts, '1') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '2', 5, 5, 64, 128, 1, 2) ;
net = add_norm(net, opts, '2') ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;


net = add_block(net, opts, '3', 3, 3, 128, 256, 1, 1) ;
%net = add_block(net, opts, '4', 3, 3, 256, 512, 1, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;


%net = add_block(net, opts, '5', 3, 3, 512, 512, 1, 1) ;
%net = add_block(net, opts, '6', 4, 4, 512, 2048, 1, 0) ;
net = add_block(net,opts, '4', 4, 4, 256, 512, 1,  0);
%net = add_block(net,opts, '4', 6, 16, 256, 256, 1,  0);
%net = add_dropout(net, opts, '7') ;

%net = add_block(net, opts, '8', 1, 1, 2048, 2048, 1, 0) ;
%net = add_dropout(net, opts, '9') ;

%net = add_block(net, opts, '10', 1, 1, 2048, 500, 1, 0) ;
%net = add_block(net, opts, '10', 1, 1, 2048, 10000, 1, 0) ;
net = add_block(net,opts, '5', 1, 1 ,512, 710, 1, 0);
%net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s','11'));
net.layers(end) = [] ;
if opts.batchNormalization, net.layers(end) = [] ; end

% final touches
switch lower(opts.weightInitMethod)
  case {'xavier', 'xavierimproved'}
    net.layers{end}.weights{1} = net.layers{end}.weights{1} / 10 ;
end
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

% Meta parameters
net.meta.inputSize = net.meta.normalization.imageSize ;
%net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
%net.meta.normalization.interpolation = 'bicubic' ;
net.meta.normalization.averageImage = [] ;
net.meta.normalization.keepAspect = true ;
%net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.jitter = opts.jitter ;


if ~opts.batchNormalization
  %lr = logspace(-2, -4, 60) ;
  lr = logspace(-2.5, -5, 40);
else
  %lr = logspace(-1, -4, 20) ;
  lr = logspace(-2.5, -5, 40);
end


net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.batchSize = 64 ;
net.meta.trainOpts.weightDecay = 0.0005 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1err') ;
    net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
                                       'opts', {'topK',5}), ...
                 {'prediction','label'}, 'top5err') ;
  otherwise
    assert(false) ;
end


% Meta parameters
%net.meta.inputSize = [48 128 1] ;
%net.meta.trainOpts.learningRate = [0.005*ones(1,30) 0.0005*ones(1,10)] ;
%net.meta.trainOpts.weightDecay = 0.0001 ;
%net.meta.trainOpts.batchSize = 64 ;

% Fill in default values
%net = vl_simplenn_tidy(net) ;
%
%% Switch to DagNN if requested
%switch lower(opts.networkType)
%  case 'simplenn'
%    % done
%  case 'dagnn'
%    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
%    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
%             {'prediction','label'}, 'error') ;
%  otherwise
%    assert(false) ;
%end
%
