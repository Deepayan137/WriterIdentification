function topScoreCalc(imageFolder,outFolder)
load([outFolder,'knn.mat']);
var = outFolder;
qTest = zeros(1:200,1:10);
dist = 1-D;
%d = dir('/media/deepayan/96AA0549AA0526F92/1/Dataset/ICDAR13/test/*.tif');
d = dir([imageFolder '*.tif']);
for i = 1:189
    for j = 2:10
        query = str2num(strtok(d(Idx(i,1)).name,'-'));
        retreived = str2num(strtok(d(Idx(i,j)).name,'-'));
        if  query == retreived
            qTest(i,j) = 1;
            
        else
            qTest(i,j) = -1;
        end
    end
end
top1 =0;top2 =0;top3 =0;
for m = 1:189
         if qTest(m,2) == 1
             top1 = top1+1;
            end
         if qTest(m,3) == 1 && qTest(m,2) == 1
            top2 = top2+1;
            end
        if qTest(m,4) == 1 && qTest(m,3) == 1 && qTest(m,2) == 1
             top3 = top3 +1;
             
         end
end


top1 = (top1/189)*100;
disp(top1);
top2 = (top2/189)*100;
disp(top2);   
top3 = (top3/189)*100;
disp(top3);        
            
        
    
vl_pr(qTest(1:189,2:10), dist(1:189,2:10)); 
