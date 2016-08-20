function knn(imageFolder,outFolder)
name = 'knnicdarSV.mat';
var = imageFolder;
pathname = strcat([outFolder,name]);
X = load([outFolder,'superVector1.mat']);
y = load([outFolder,'superVector1.mat']);
%X = load('fisher_query_epoch20.mat');
%y= load('fisher_query_epoch20.mat');
%[XCOEFF,XSCORE] = princomp(X.enc);
%[yCOEFF,ySCORE] = princomp(y.enc);
%z = load('/media/deepayan/96AA0549AA0526F92/1/Dataset/ICDAR13/test/data_batch_1.mat.mat');
%d = dir('/media/deepayan/96AA0549AA0526F92/1/Dataset/ICDAR13/*.tif');
IDX= [];
[Idx,D]=knnsearch(abs(X.SV),abs(y.SV),'K',1000,'distance','cosine');
for i = 1:size(Idx,1)
 str = sprintf('%d',i);
 %fid = fopen(['/media/deepayan/96AA0549AA0526F92/1/Dataset/ICDAR13/',str],'w');
 
 n =transpose(Idx(i,:));
 m = transpose(D(i,:));
 %for j = 1:size(Idx,2)
 %    str = strcat('/media/deepayan/96AA0549AA0526F92/1/Dataset/ICDAR13/patches2_test/',d(n(j)).name);
 %    imshow(str);
 %  fprintf(fid,'%s:%d\n',d(n(j)).name,m(j));
 %end
 
end
save(pathname,'Idx','D');
