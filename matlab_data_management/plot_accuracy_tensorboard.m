step = [];train_acc = [];test_acc = [];val_acc = [];
train_cost = [];test_cost = [];val_cost = [];

step = [step;trainaccuracyaccuracy.step];
train_acc = [train_acc;trainaccuracyaccuracy.value];
test_acc = [test_acc;testaccuracyaccuracy.value];
val_acc = [val_acc;validationaccuracyaccuracy.value];
train_cost = [train_cost;traincostcost.value];
test_cost = [test_cost;testcostcost.value];
val_cost = [val_cost;validationcostcost.value];

%%
figure('name','1','rend','painters','pos',[50 400 1800 500]);
hold on; box on; grid on;
sbp = subplot(1,1,1);
yyaxis left
plot(step, smooth(train_acc,100),'LineWidth',1.2)
plot(step, smooth(val_acc,100),'LineWidth',1.2)
plot(step, smooth(test_acc,100),'LineWidth',1.2)

sbp.YLim = [0 1.1];
sbp.XLim = [0 10*100000];
sbp.LineWidth = 1.2;
sbp.FontSize = 20;
xlabel('Steps');
ylabel('Accuracy');

yyaxis right
plot(step, smooth(train_cost,100),'LineWidth',1.2)
plot(step, smooth(val_cost,100),'LineWidth',1.2)
plot(step, smooth(test_cost,100),'LineWidth',1.2)
sbp.YLim = [0 2];
ylabel('Loss');
lgd = legend({'Training Accuracy','Validation Accuracy','Testing Accuracy',...
    'Training Loss','Validation Loss','Testing Loss'},'Location','east');
lgd.FontSize = 20;

%%

c1 = traincostcost.wall_time(train_cost <= 1);
s1 = step(train_cost <= 1);
s1 = s1(1);
c1 = (c1(1)-traincostcost.wall_time(1))/60;
c01 = traincostcost.wall_time(train_cost <= 0.1);
s01 = step(train_cost <= 0.1);
s01 = s01(1);
c01 = (c01(1)-traincostcost.wall_time(1))/60;
c001 = traincostcost.wall_time(train_cost <= 0.01);
s001 = step(train_cost <= 0.01);
s001 = s001(1);
c001 = (c001(1)-traincostcost.wall_time(1))/60;

s1
c1
s01
%%
c01
s001
c001



%%

step(10001:end) = step(10001:end)+step(10000);

