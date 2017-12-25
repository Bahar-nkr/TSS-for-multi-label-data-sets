function [er] = MyCost(X,train,train_labels)
index=find(X~=0);
DT=train(index,:);
Train_labels=train_labels(index,:);
% class=knnclassify(train,DT,Train_labels,5);
% ACC=(sum(class==train_labels))./length(class);
Num=10;
Smooth=1;

[Prior,PriorN,Cond,CondN]=MLKNN_train(DT,Train_labels,Num,Smooth);        
[output,predict_label]=MLKNN_test(DT,Train_labels,train,Num,Prior,PriorN,Cond,CondN);
[acc,Pre,rec,F_mes] = PrecisionRecall(predict_label,train_labels);
er=1-acc;

end

