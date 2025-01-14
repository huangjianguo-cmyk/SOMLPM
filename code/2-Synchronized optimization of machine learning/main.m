%% 基于ITSO的同步优化-Ligitgbm

%% 清空变量 
clc;clear                                 % 清空变量
close all                                 % 关闭窗口 

%% 读取数据
data=xlsread("data.xlsx");
input0=data(:,1:end-1);                   % 训练特征（自变量）
output=data(:,48);                        % 输出变量（因变量）
Num=length(output);                       % 计算样本量
Num_d=size(input0,2);                     % 特征数量
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

%% 特征筛选+超参数同步优化[SVM]
SearchAgents_no=30;              % 种群规模
Max_iteration=100;                % 最大迭代次数
dim=Num_d+2;                     % 自变量维度=特征数量+超参数个数
lb= zeros(1,dim);                % 自变量下界
ub= ones(1,dim);                 % 自变量上界
% SVM模型超参数寻优范围
lb(1)=0.001;
ub(1)=1000;
lb(2)=0.001;
ub(2)=1000;
fobj=@(x) OBJ_ligGBM(x,XTrain,YTrain);  % 设置目标函数
% 改进策略设置：
num1 = 5;  % Piecewise
num2 = 3;  % 自适应t分布
num3 = 10; % 黄金正弦
[Best_score,Best_pos,cg_curve]=TSO(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

% 结果汇总
x_f=Best_pos(3:end);
x_f(:,x_f<0.5)=0;
find(x_f~=0)
param1=Best_pos(1);
param2=Best_pos(2);
rng(1)          % 为了重现性
% 模型训练
mdl1=fitcgbm(XTrain,YTrain, ...
            'KernelFunction','RBF',...
            'KernelScale',param1 , ...
            'BoxConstraint', param2, ...
            'Standardize', true, ...
            'ClassNames', [0; 1] );
% 模型测试
[validationPredictions,validationScores]=predict(mdl1,XTest);
[x1,y1,~,auc1] = perfcurve(YTest,validationScores(:,2),'1');
figure(1)
rocObj1 = rocmetrics(YTest,validationScores,{'0';'1'});
plot(rocObj1,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)

figure(2)
CNT=20;   % 绘制图形随机选点的数量(可自行设置)
k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
iter=1:1:Max_iteration;
semilogy(iter(k),cg_curve(k),'-*','LineWidth',1)
grid off
xlabel('Iteration')
ylabel('Fitness')
title('Iteration curve')
axis tight
grid on
box on
set(gca,'FontName','Times New Rome','FontSize',10);

%% 特征筛选+超参数同步优化[LR]
SearchAgents_no=30;                  % 种群规模
Max_iteration=100;                   % 最大迭代次数
dim=Num_d;                           % 自变量维度=特征数量+超参数个数
lb= zeros(1,dim);                    % 自变量下界
ub= ones(1,dim);                     % 自变量上界
fobj=@(x) OBJ_lr(x,XTrain,YTrain);  % 设置目标函数
% 改进策略设置：
num1 = 5;  % Piecewise
num2 = 3;  % 自适应t分布
num3 = 10; % 黄金正弦
[Best_score,Best_pos,cg_curve]=TSO(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

% 结果汇总
x_f=Best_pos(1:end);
x_f(:,x_f<0.5)=0;
find(x_f~=0)
rng(1)          % 为了重现性
% 模型训练
mdl1 = fitglm(...
            XTrain,YTrain, ...
            'Distribution', 'binomial', ...
            'link', 'logit');
% 模型测试
[validationPredictions,validationScores]=predict(mdl1,XTest);
[x1,y1,~,auc1] = perfcurve(YTest,validationScores(:,2),'1');
figure(1)
rocObj1 = rocmetrics(YTest,validationScores,{'0';'1'});
plot(rocObj1,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)

figure(2)
CNT=20;   % 绘制图形随机选点的数量(可自行设置)
k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
iter=1:1:Max_iteration;
semilogy(iter(k),cg_curve(k),'-*','LineWidth',1)
grid off
xlabel('Iteration')
ylabel('Fitness')
title('Iteration curve')
axis tight
grid on
box on
set(gca,'FontName','Times New Rome','FontSize',10);

%% 特征筛选+超参数同步优化[tree]
SearchAgents_no=30;                  % 种群规模
Max_iteration=100;                   % 最大迭代次数
dim=Num_d+1;                           % 自变量维度=特征数量+超参数个数
lb= zeros(1,dim);                    % 自变量下界
ub= ones(1,dim);                     % 自变量上界
lb(1)=1;
ub(1)=40;
fobj=@(x) OBJ_tree(x,XTrain,YTrain);  % 设置目标函数
% 改进策略设置：
num1 = 5;  % Piecewise
num2 = 3;  % 自适应t分布
num3 = 10; % 黄金正弦
[Best_score,Best_pos,cg_curve]=TSO(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

% 结果汇总
x_f=Best_pos(2:end);
x_f(:,x_f<0.5)=0;
find(x_f~=0)
rng(1)          % 为了重现性
param1=Best_pos(1);
param1=round(param1);
% 模型训练
mdl1 = fitctree(...
            XTrain,YTrain, ...
            'MaxNumSplits', param1, ...
            'Surrogate', 'off', ...
            'ClassNames', [0; 1]);
% 模型测试
[validationPredictions,validationScores]=predict(mdl1,XTest);
[x1,y1,~,auc1] = perfcurve(YTest,validationScores(:,2),'1');
figure(1)
rocObj1 = rocmetrics(YTest,validationScores,{'0';'1'});
plot(rocObj1,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)



figure(2)
CNT=20;   % 绘制图形随机选点的数量(可自行设置)
k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
iter=1:1:Max_iteration;
semilogy(iter(k),cg_curve(k),'-*','LineWidth',1)
grid off
xlabel('Iteration')
ylabel('Fitness')
title('Iteration curve')
axis tight
grid on
box on
set(gca,'FontName','Times New Rome','FontSize',10);

%% 特征筛选+超参数同步优化[BP]
SearchAgents_no=30;              % 种群规模
Max_iteration=100;                % 最大迭代次数
dim=Num_d+2;                     % 自变量维度=特征数量+超参数个数
lb= zeros(1,dim);                % 自变量下界
ub= ones(1,dim);                 % 自变量上界
% SVM模型超参数寻优范围
lb(1)=1;
ub(1)=100;
lb(2)=0.001;
ub(2)=15;
fobj=@(x) OBJ_bp(x,XTrain,YTrain);  % 设置目标函数
% 改进策略设置：
num1 = 5;  % Piecewise
num2 = 3;  % 自适应t分布
num3 = 10; % 黄金正弦
[Best_score,Best_pos,cg_curve]=TSO(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);

% 结果汇总
x_f=Best_pos(3:end);
x_f(:,x_f<0.5)=0;
find(x_f~=0)
param1=Best_pos(1);
param1=round(param1);
param2=Best_pos(2);
rng(1)          % 为了重现性
% 模型训练
 mdl1 = fitcnet(...
            XTrain,YTrain, ...
            'LayerSizes', param1, ...
            'Lambda', param2, ...
            'IterationLimit', 1000, ...
            'Standardize', true, ...
            'ClassNames', [0; 1]);
% 模型测试
[validationPredictions,validationScores]=predict(mdl1,XTest);
[x1,y1,~,auc1] = perfcurve(YTest,validationScores(:,2),'1');
figure(1)
rocObj1 = rocmetrics(YTest,validationScores,{'0';'1'});
plot(rocObj1,"ClassNames","1",ShowModelOperatingPoint=false,LineWidth=1.5)

figure(2)
CNT=20;   % 绘制图形随机选点的数量(可自行设置)
k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
iter=1:1:Max_iteration;
semilogy(iter(k),cg_curve(k),'-*','LineWidth',1)
grid off
xlabel('Iteration')
ylabel('Fitness')
title('Iteration curve')
axis tight
grid on
box on
set(gca,'FontName','Times New Rome','FontSize',10);

