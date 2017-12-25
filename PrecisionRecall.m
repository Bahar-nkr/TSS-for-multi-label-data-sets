
function [accuracy,Precision,Recall,F_measure] = PrecisionRecall(predict_label,test_target)%returns average precisions & recalls of all classes
predict_label=predict_label';
test_target=test_target';
[~,num_testing]=size(predict_label);
p=0;f=0;r=0;ac=0;
for i=1:num_testing
    a=find(predict_label(:,i)==1);
    b=find(test_target(:,i)==1);
    nom=length(intersect(a,b));
    nom1=length(union(a,b));
    predict_length=length(a);
    test_length=length(b);
    p=p+nom/(predict_length+eps);
    r=r+nom/(test_length+eps);
    f=f+(2*nom)/(predict_length+test_length);
    ac=ac+nom/nom1;
end
Precision=p/(num_testing);
Recall=r/(num_testing);
F_measure=f/num_testing;
accuracy=ac/num_testing;
end

