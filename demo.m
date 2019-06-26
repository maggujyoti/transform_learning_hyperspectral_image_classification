% Prepare Dataset

clc; clear; close all;

datasetName='IndianPines'; param.percent=40; param.features=false;
%[img,gt,X,numCls,trnIdx,tstIdx]=funLoadClsData(datasetName, percent);
[img,gt,X,numCls,trnIdx,tstIdx,numTrainEachCls,numTestEachCls]=funLoadClsData(datasetName, param);
[m,n,d]=size(img);
trnX=X.trnX;
trnY=X.trnY;
tstX=X.tstX;
tstY=X.tstY;
nTst=numel(tstIdx);

% Apply Algorithm
tic;
numAtom=100;epsilon=.05; lambda=50;mu=1;
[T, Z, W] =lcTransformLearning(trnX',trnY,numAtom,epsilon,lambda,mu);
toc;
tic;
Z_test = T*tstX';
Label_test_pred= W * Z_test;
[~, pred] = max(Label_test_pred);
pred=pred';
toc;
%for accuracy
cm=confusionmat(X.tstY,pred);
OA=(sum(diag(cm))/sum(cm(:)))*100; 
CA=(diag(cm)./sum(cm,2))*100; 
AA=mean(CA); %This is accuracy
cip=sum(cm,2);cpi=sum(cm)';Pe= sum(cip.*cpi)/(nTst^2);
kappa= (.01*OA-Pe)/(1-Pe);
fprintf('OA=%1.2f AA=%1.2f kappa=%1.3f \n',OA,AA,kappa);

