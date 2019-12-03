far = [];
frr = [];
eer_all = [];

%%

results = [labels.VarName1,predictions.VarName1];
frr_mat = zeros(109,1);
far_mat = zeros(109,1);

for s=0:108
    resultsTempTure = results(results(:,1)==s,:);
    resultsTempFalse = results(results(:,2)==s,:);
    cnt4 = size(resultsTempFalse,1);
    resultsTempFalse(resultsTempFalse(:,1)==s,:)=[];
    cnt3 = size(resultsTempFalse,1);
    % FRR
    cnt = 0;cnt1 = 0;
    for i=1:size(resultsTempTure,1)
        cnt1 = cnt1+1;
        if resultsTempTure(i,1)~=resultsTempTure(i,2)
            cnt = cnt+1;
        end
    end
    frrTemp = cnt/cnt1;
    frr_mat(s+1) = frrTemp;
    
    % FAR
    farTemp = cnt3/cnt4
    far_mat(s+1) = farTemp;
    
end

FAR_mean = mean(far_mat)
FRR_mean = mean(frr_mat)
eer = (mean(frr_mat)+mean(far_mat))/2

far = [far,FAR_mean];
frr = [frr,FRR_mean];
eer_all = [eer_all,eer];

%% ROC Curve

hold on;
for s=0
    resultsTemp = results;
%     resultsTemp(results(:,1)==s,1) = 1;
%     resultsTemp(results(:,1)~=s,1) = -1;
%     resultsTemp(results(:,2)==s,2) = 1;
%     resultsTemp(results(:,2)~=s,2) = -1;
    
    %[X,Y] = perfcurve(resultsTemp(:,1),resultsTemp(:,2),s,'XCrit', 'fpr');
    [auc, ll, X, Y] = colAUC(resultsTemp(:,2),resultsTemp(:,1),'ROC','plot',true);
end
%plot(X,Y)

%%
XX = X;
for i=1:5886
    C = unique(XX(:,i));
    for c=1:length(C)
        [B,I] = find(XX(:,i)==C(c));
        if C(c) ~= 0
            for j=1:length(B)
                XX(B(j),i) = XX(B(j),i)-0.0000001*j;
            end
        else
            for j=1:length(B)
                XX(B(j),i) = XX(B(j),i)+0.0000001*(length(B)-j);
            end
        end
    end
end

%%

Y_new = zeros(5886,100);

for i=1:5886
    Y_new(i,:) = interp1(XX(:,i),Y(:,i),0:0.01:(1-0.01));
end
Y_new = sum(Y_new)/5886;
figure;hold on
plot(Y_new)
save('cldnn_3','Y_new')
