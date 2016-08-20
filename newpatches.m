function newpatches(imageFolder,outFolder)
d = dir([imageFolder '*.mat']);
nfiles = length(d);
TaskIDStr = getenv('SLURM_ARRAY_TASK_ID');
StepIDStr = getenv('CUSTOM_ARRAY_STEP_ID');
fprintf('slurm%s step %s\n',TaskIDStr,StepIDStr);
if ~isempty(TaskIDStr) && ~isempty(StepIDStr)
    startIdx = (str2double(TaskIDStr)-1) * str2double(StepIDStr) + 1;
    endIdx = startIdx+str2double(StepIDStr)-1;
    i = (str2double(TaskIDStr));
else
    disp('No task ID specified');
    startIdx =6;
    endIdx = 10;
    i =1;
end
%str = sprintf('data_batch_%d.mat',i);
%pathname = strcat([outFolder,str]);
count = 0;level = 0.5;
%data =[];labels=[];
for l = startIdx:endIdx
    data =[];labels=[];
    str = sprintf('data_batch_%d.mat',l);
    pathname = strcat([outFolder,str]);
    %data = zeros(4000,1024);
    %labels = zeros(4000,1);
    count = count+1;
    fname =strcat(num2str(l),'.mat');
    load(strcat(imageFolder,fname));
    siz = size(fontData,2);
truj = 0;
 for j = 1:siz
        image = fontData{j};
    if (size(image,2)) >= 100 
     if size(find(~im2bw(image)),1) >=1000
        image = padarray(image,[100,100],'replicate');
        [x, y, z] = size(image);
        truj = truj+1;
        x = x-100;
        y = y-100;
        C = 0;
        disp(sprintf('%s-%d',fname,j));
        cell = [31 47 63];
        k=0;
	S = size(data,1);
        while(k<1000)
            k = k+1;
            ran = randperm(3);
            num = cell(ran);
            x1 = randi([1,x],1,1);
            y1 = randi([1,y],1,1);
            new_img = image(x1:x1+num(1),y1:y1+num(1));
            bimg = im2bw(new_img);
            nz= (find(bimg));
            z = (find(~bimg));
            im_ar = new_img;
            
            if size(z,1) > 300
                C = C+1;
                %imshow(im_ar);
                im_ar = imresize(im_ar,[32 32]);
		if(islogical(im_ar))
      		 [r c] = find(~im_ar);
      		 im_ar = im_ar(min(r):max(r),min(c):max(c));
   		 end
   		 im_ar = single(im_ar);
   		 s = std(im_ar(:));
       		 im_ar = im_ar - mean(im_ar(:));
       		 im_ar = im_ar / ((s + 0.0001) / 128.0);
		
                imag = reshape(im_ar,1,[]);
                %imag = double(imag);
                b = S+C;
                data(b,:) = imag;
                labels(b,:) = str2num(strtok(fname,'.mat'));
                if C >=4
                    break;
                end
            end
        end
      end
   end
 end
save(pathname,'data','labels','-v7.3');
end
disp('saved');
%save(pathname,'data','labels','-v7.3');
