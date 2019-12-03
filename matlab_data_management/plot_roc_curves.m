
c=load('cldnn_1.mat');cldnn(1,:) = c.Y_new;
c=load('cldnn_2.mat');cldnn(2,:) = c.Y_new;
c=load('cldnn_3.mat');cldnn(3,:) = c.Y_new;
c=load('cnn_1.mat');cnn(1,:) = c.Y_new;
c=load('cnn_2.mat');cnn(2,:) = c.Y_new;
c=load('cnn_3.mat');cnn(3,:) = c.Y_new;
c=load('lstm_1.mat');lstm(1,:) = c.Y_new;
c=load('lstm_2.mat');lstm(2,:) = c.Y_new;
c=load('lstm_3.mat');lstm(3,:) = c.Y_new;

%%

figure('name','1','rend','painters','pos',[50 400 650 500]);
hold on; box on; grid on;
sbp = subplot(1,1,1);
errorbar(0:0.01:0.99,mean(cldnn),std(cldnn),'-s','LineWidth',1.2)
errorbar(0:0.01:0.99,mean(cnn),std(cnn),'-o','LineWidth',1.2)
errorbar(0:0.01:0.99,mean(lstm),std(lstm),'-^','LineWidth',1.2)
xlim([0 0.2])
sbp.LineWidth = 1.2;
sbp.FontSize = 20;

xlabel('False positive rate') 
ylabel('True positive rate')
legend('Proposed','CNN','LSTM','Location','Best')
lgd.FontSize = 20;