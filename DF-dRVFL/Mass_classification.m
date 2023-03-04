% clear;
load data_partition.mat;
load Trained_vgg19_0226_2.mat;
model_name='dRVFL';
date='0421';
partStores{1} = imd1.Files ;
partStores{2} = imd2.Files ;
partStores{3} = imd3.Files ;
partStores{4} = imd4.Files ;
partStores{5} = imd5.Files ;

Nlayer=[6 12 18 24];
% eval(['diary training_' model_name '_' date '_train_DDSM_test_DDSM.txt;']);
inputSize = trainedNet{1,1}.Layers(1).InputSize;
% eval(['diary training_' model_name '_' date '_train_DDSM_test_DDSM.txt;']);

% [imd1,imd2,imd3,imd4,imd5]= splitEachLabel(imds,0.2,0.2,0.2,0.2);
idx=[1,2,3,4,5];
activation_layer = 'fc64';
% N = [40,80,100,200,300,400,500,80,1000,1200,1400,1600,1800,2000];
for z=1:size(Nlayer,2)
eval(['diary training_' model_name '_' date '_' activation_layer '_Nlayer_' num2str(Nlayer(1,z)) '.txt;']);
for i = 1:5
    i
    test_idx = (idx == i);
    train_idx = ~test_idx;
    imdsTest = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true,'FileExtensions','.png', 'LabelSource', 'foldernames');
    imdsTrain = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'FileExtensions','.png', 'LabelSource', 'foldernames');
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing','gray2rgb');
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsTest,'ColorPreprocessing','gray2rgb');
 
      
    trainy = imdsTrain.Labels;
    trainy=double(trainy);
    testy = imdsTest.Labels;
    testy=double(testy);

    trainx = activations(trainedNet{i,1},augimdsTrain,activation_layer,'OutputAs','rows');
    testx = activations(trainedNet{i,1}, augimdsValidation,activation_layer, 'OutputAs','rows');

    tic, net{i}=dRVFLtrain(trainx, trainy, [Nlayer(1,z) Nlayer(1,z) Nlayer(1,z)]);toc
    [y{i},scores{i}]=dRVFLtest(testx, net{i});
    
%     Accuracy{i}=mean(testy==y{i});
    cfmdRVFL(:,:,i)= confusionmat(y{i},testy);
    [sensitivity(i),specificity(i),accuracy(i),precision(i),F1(i)] = getindexes(cfmdRVFL(:,:,i));
end
eval(['save ' 'results_of_' model_name '_date_' date '_' activation_layer '_Nlayer_' num2str(Nlayer(1,z)) ' y scores sensitivity  specificity accuracy  precision F1 cfmdRVFL;']);
eval(['save ' 'trained_' model_name '_date_' date '_' activation_layer '_Nlayer_' num2str(Nlayer(1,z)) ' net;']);

diary off;
end 
% eval(['save ' 'results_of_vgg19_12_epoches_' activation_layer '_N_' num2str(N(1,z)) ' cfmelm cfmsnn cfmrvfl cfmen TestScores1 TestScores2 TestScores3;']);
% if z==1
%     save Trained_vgg19.mat YPred Scores YValidation accuracy confMat;
% end
% end
% toc




% yypred=double(TTest);
% yylabel = double(YValidation);
% result=AUC(yylabel,yypred);
% yylabel_1 = ind2-1;
% yypred_1 = yypred-1;
% yylabel = double(TTest);
% yylabel_1 = yylabel-1;
% yypred_1 = ind2-1
% ROC = v2plot_roc(yypred_1, yylabel_1);
% auc = AUC(yylabel_1,yypred_1);
% 
% m_acc=mean(accuracy_net)
% m_sen=mean(sensitivity_net)
% m_spe=mean(specificity_net)
% m_pre=mean(precision_net)
% m_F1=mean(F1_net)