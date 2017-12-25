clear all;clc; close all
min_flag=0;
load('flag.mat');
dataset_name = 'flag_ML';
ktimes = 5; % 5FCV,fixed
file_name = dataset_name;
dataname =  strcat('BBA_MLknn_',file_name);
final_res = zeros(ktimes+2,8);
targets(find(targets==-1))==1;
indices = crossvalind('kfold',size(data,1),5);
for i_cv = 1:5
    i_cv
    test_ind = (indices==i_cv);test=data(test_ind,:);test_labels=targets(test_ind,: );
    train_ind = ~test_ind;train=data(train_ind,:);train_labels=targets(train_ind,:);
    
    Max_iteration=100; % Maximum number of iterations
    noP=50; % Number of particles
    noV=size(train,1);
    
    A=.25;      % Loudness  
    r=.1;      % Pulse rate 
    for ij=1:6
        %%%%%%%%%%%% BBA %%%%%%%%%%%%%
        [gBest, gBestScore]=BBA(noP, A, r, noV, Max_iteration,train,train_labels);
        ACC_train(i_cv,ij)=1-gBestScore;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        indexx=find(gBest~=0);
        Reduction(i_cv,ij)=((noV-length(indexx))./noV).*100;
        DT=train(indexx,:);
        Train_labels=train_labels(indexx,:);
        Num=10;
        Smooth=1;
        [Prior,PriorN,Cond,CondN]=MLKNN_train(DT,Train_labels,Num,Smooth);
        [output,predict_label]=MLKNN_test(DT,Train_labels,test,Num,Prior,PriorN,Cond,CondN);
        [acc(i_cv,ij),Pre(i_cv,ij),rec(i_cv,ij),F_mes(i_cv,ij)] = PrecisionRecall(predict_label,test_labels);
        HLoss(i_cv,ij)=Hamming_loss(predict_label,test_labels);
        RLoss(i_cv,ij)=Ranking_loss(output,test_labels);
    end
    final_res(i_cv,1) = mean(ACC_train(i_cv,:));
    final_res(i_cv,2) = mean(acc(i_cv,:));
    final_res(i_cv,3) = mean(Reduction(i_cv,:));
    final_res(i_cv,4) = mean(Pre(i_cv,:));
    final_res(i_cv,5) = mean(rec(i_cv,:));
    final_res(i_cv,6) = mean(F_mes(i_cv,:));
    final_res(i_cv,7) = mean(HLoss(i_cv,:));
    final_res(i_cv,8) = mean(RLoss(i_cv,:));
end
final_res(ktimes+1,:) = mean(final_res(1:ktimes,:));
final_res(ktimes+2,:) = std(final_res(1:ktimes,:));
save([dataname,'.mat'],'final_res');
