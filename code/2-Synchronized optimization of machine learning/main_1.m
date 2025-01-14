%% 清空变量 
clc;clear                                 % 清空变量
close all                                 % 关闭窗口 

%% 读取数据
data=xlsread("lasso.xlsx");
input0=data(:,1:end-1);                   % 训练特征（自变量）
output=data(:,9);                        % 输出变量（因变量）
Num=length(output);                       % 计算样本量

%% 切分数据集
c = cvpartition(output,"HoldOut",0.2);    % 数据集切分比例设置（可以调整数值）
trainingIndices = training(c);            % 训练集索引
testIndices = test(c);                    % 测试集索引
XTrain = input0(trainingIndices,:);       % 训练集_训练特征（自变量）
YTrain = output(trainingIndices);         % 训练集_输出变量（因变量）
D_train=[YTrain XTrain];
Num_trian=length(YTrain);                 % 获取训练集的样本量
XTest = input0(testIndices,:);            % 测试集_训练特征（自变量）
YTest = output(testIndices);              % 测试集_输出变量（因变量）
D_test=[YTest XTest];




%% 测试集验证
[validationPredictions1,validationScores1]=trainedModel1.predictFcn(XTest); 
[validationPredictions2,validationScores2]=trainedModel2.predictFcn(XTest); 
[validationPredictions3,validationScores3]=trainedModel3.predictFcn(XTest); 
[validationPredictions4,validationScores4]=trainedModel4.predictFcn(XTest); 
[validationPredictions5,validationScores5]=trainedModel5.predictFcn(XTest); 
[validationPredictions6,validationScores6]=trainedModel6.predictFcn(XTest); 
[validationPredictions7,validationScores7]=trainedModel7.predictFcn(XTest); 

%% 结果测试
[x1,y1,~,auc1] = perfcurve(YTest,validationScores1(:,2),'1');
[x2,y2,~,auc2] = perfcurve(YTest,validationScores2(:,2),'1');
[x3,y3,~,auc3] = perfcurve(YTest,validationScores3(:,2),'1');
[x4,y4,~,auc4] = perfcurve(YTest,validationScores4(:,2),'1');
[x5,y5,~,auc5] = perfcurve(YTest,validationScores5(:,2),'1');
[x6,y6,~,auc6] = perfcurve(YTest,validationScores6(:,2),'1');
[x7,y7,~,auc7] = perfcurve(YTest,validationScores7(:,2),'1');

%% 交叉验证结果
% 诊断指标（1）
C = confusionmat(YTest,validationPredictions1);  % 先计算训练集混淆矩阵
stats1 = statsOfMeasure(C);%训练集指标
% 诊断指标（2）
C = confusionmat(YTest,validationPredictions2);  % 先计算训练集混淆矩阵
stats2 = statsOfMeasure(C);%训练集指标
% 诊断指标（3）
C = confusionmat(YTest,validationPredictions3);  % 先计算训练集混淆矩阵
stats3 = statsOfMeasure(C);%训练集指标
% 诊断指标（4）
C = confusionmat(YTest,validationPredictions4);  % 先计算训练集混淆矩阵
stats4 = statsOfMeasure(C);%训练集指标
% 诊断指标（5）
C = confusionmat(YTest,validationPredictions5);  % 先计算训练集混淆矩阵
stats5 = statsOfMeasure(C);%训练集指标
% 诊断指标（6）
C = confusionmat(YTest,validationPredictions6);  % 先计算训练集混淆矩阵
stats6 = statsOfMeasure(C);%训练集指标
% 诊断指标（7）
C = confusionmat(YTest,validationPredictions7);  % 先计算训练集混淆矩阵
stats7 = statsOfMeasure(C);%训练集指标

writetable(stats1,'结果.xlsx',  'Sheet',1);
writetable(stats2,'结果.xlsx',  'Sheet',2);
writetable(stats3,'结果.xlsx',  'Sheet',3);
writetable(stats4,'结果.xlsx',  'Sheet',4);
writetable(stats5,'结果.xlsx',  'Sheet',5);
writetable(stats6,'结果.xlsx',  'Sheet',6);
writetable(stats7,'结果.xlsx',  'Sheet',7);
%% ROC曲线

rocObj1 = rocmetrics(YTest,validationScores1,{'0';'1'});
plot(rocObj1,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)
hold on
rocObj2 = rocmetrics(YTest,validationScores2,{'0';'1'});
plot(rocObj2,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)
hold on
rocObj3 = rocmetrics(YTest,validationScores3,{'0';'1'});
plot(rocObj3,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)
hold on
rocObj4 = rocmetrics(YTest,validationScores4,{'0';'1'});
plot(rocObj4,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)
hold on
rocObj5 = rocmetrics(YTest,validationScores4,{'0';'1'});
plot(rocObj5,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)
hold on
rocObj6 = rocmetrics(YTest,validationScores4,{'0';'1'});
plot(rocObj6,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)
hold on
rocObj7 = rocmetrics(YTest,validationScores4,{'0';'1'});
plot(rocObj7,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)
legend(['DT-AUC=',num2str(auc1)], ...
    ['LDA-AUC=',num2str(auc2)], ...
    ['LR-AUC=',num2str(auc3)], ...
    ['NB-AUC=',num2str(auc4)],...
    ['SVM-AUC=',num2str(auc5)], ...
    ['KNN-AUC=',num2str(auc6)], ...
    ['BPNN-AUC=',num2str(auc7)], ...
    Location="southeast")
title('ROC Curve'); 
set(gca,'FontName','Times New Rome','Box','on','FontSize',12,'LineWidth',1.5);


%% PR曲线
curveObj = plot(rocObj1,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate" ...
    ,LineWidth=1.5);
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc1 = trapz(xyData(:,1),xyData(:,2));
hold on
curveObj = plot(rocObj2,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate" ...
    ,LineWidth=1.5);
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc2 = trapz(xyData(:,1),xyData(:,2));
hold on
curveObj = plot(rocObj3,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate" ...
    ,LineWidth=1.5);
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc3 = trapz(xyData(:,1),xyData(:,2));
hold on
curveObj = plot(rocObj4,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate" ...
    ,LineWidth=1.5);
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc4 = trapz(xyData(:,1),xyData(:,2));
hold on
curveObj = plot(rocObj5,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate" ...
    ,LineWidth=1.5);
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc5 = trapz(xyData(:,1),xyData(:,2));
hold on
curveObj = plot(rocObj6,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate" ...
    ,LineWidth=1.5);
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc6 = trapz(xyData(:,1),xyData(:,2));
hold on
curveObj = plot(rocObj7,ClassNames='1', ...
    YAxisMetric="PositivePredictiveValue",XAxisMetric="TruePositiveRate" ...
    ,LineWidth=1.5);
xyData = rmmissing([curveObj.XData curveObj.YData]);
auc7 = trapz(xyData(:,1),xyData(:,2));
hold on
xlabel('真阳性率');  
ylabel('阳性预测值');  
legend('DT-AUC=0.99663', ...
    ['LDA-AUC=',num2str(auc2)], ...
    ['LR-AUC=',num2str(auc3)], ...
    ['NB-AUC=',num2str(auc4)],...
    ['SVM-AUC=',num2str(auc5)], ...
    ['KNN-AUC=',num2str(auc6)], ...
    ['BPNN-AUC=',num2str(auc7)], ...
    Location="southwest")
title("PR 曲线")
set(gca,'FontName','Times New Rome','Box','on','FontSize',12,'LineWidth',1.5);
