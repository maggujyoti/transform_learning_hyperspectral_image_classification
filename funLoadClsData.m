function [img,gt,X,numCls,trnIdx,tstIdx,numTrainEachCls,numTestEachCls]=funLoadClsData(datasetName, param)

 %         
 percent=param.percent;
 features=param.features;

 %gt stands for GroundTruth i.e. actual class labels
 if strcmp(datasetName,'IndianPines')
     load('Indian_pines_corrected');
     load('Indian_pines_gt');
     img=indian_pines_corrected;
     gt=indian_pines_gt;  
 elseif strcmp(datasetName,'PaviaU')
     load('PaviaU')
     load('PaviaU_gt')
     img=paviaU;
     gt=double(paviaU_gt);  
 elseif strcmp(datasetName,'Pavia')
     load('Pavia')
     load('Pavia_gt')
     img=pavia; clear pavia;
     gt=double(pavia_gt);      
 end
[m,n,b]=size(img);

%img=myhisteq(img);
for i=1:b
    img(:,:,i)= mat2gray(img(:,:,i));
    %img(:,:,i)=im2double(img(:,:,i));% 
end

totalPixels=m*n;
data=reshape(img,m*n,b);
labels=reshape(gt,m*n,1);
clsLabels= unique(labels(labels~=0));
numCls=numel(clsLabels); %number of classes
clsIdx=clsLabels(:,ones(1,totalPixels));
labelsRepeat=labels(:,ones(1,numCls))';
EachClsIdx=(clsIdx==labelsRepeat);
percentEachClass=percent;
cnts=sum(EachClsIdx,2);
numTrainEachCls= floor(cnts*.01*percentEachClass);

if strcmp(datasetName,'IndianPines')
    numTrainEachCls([1 4 7 9 16])=[15 50 20 15  50];
end
numTestEachCls=cnts-numTrainEachCls;
nTrn=sum(numTrainEachCls);
nTst=sum(numTestEachCls);

st=1;ed=0; stTest=1; edTest=0;
trnIdx=zeros(nTrn,1);
tstIdx=zeros(nTst,1);

for i=1:numCls
    temp=find(EachClsIdx(i,:)==1);%  find(labels==i);    
    ed=ed+numTrainEachCls(i);
    edTest=edTest+numTestEachCls(i);
    trnIdx(st:ed)=temp(randperm(cnts(i),numTrainEachCls(i))); 
    tstIdx(stTest:edTest)=setdiff(temp,trnIdx(st:ed));
    st=ed+1;
    stTest=edTest+1;
end
X.trnX=data(trnIdx,:);
X.trnY=labels(trnIdx);
X.tstX=data(tstIdx,:);
X.tstY=labels(tstIdx);

 
