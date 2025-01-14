

%% 改进群体智能算法终结者
%图灵翼公司独创
clear
clc
close all

Function_name='F23'; % 测试函数编号（F1~F23）
[lb,ub,dim,fobj]=Get_Functions_details(Function_name); % 获取目标函数对应参数
SearchAgents_no=30; %种群规模
Max_iteration=1000; %最大迭代次数

%% 模型训练
num1 = 5;  % 头部混沌变异改进方案选择：1-10，tent、Logistic、Cubic、chebyshev、Piecewise、sinusoidal、Sine,ICMIC, Circle,Bernoulli
num2 = 3;  % 身体融合变异改进方案选择：1-15，高斯变异，t分布扰动，自适应t分布，柯西变异扰动，差分变异扰动，高斯随机游走，莱维飞行,三角形游走,螺旋飞行，黄金正弦，正余弦,透镜成像反向学习,纵横交叉，动态反向学习，随机游走
num3 = 10;  % 尾部拼接变异改进方法选择：与融合变异相同

% TSO-金枪鱼群优化算法
[Best_score,Best_pos,cg_curve]=TSO(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);      
num1 = 0;  % 无改进
num2 = 0;  % 无改进
num3 = 0;  % 无改进
[Best_score1,Best_pos1,cg_curve1]=TSO(num1,num2,num3,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);  

%% 结果绘图
figure('Position',[500 500 660 290])
% 绘制搜索空间
subplot(1,2,1);
func_plot(Function_name);
title('Test function')
xlabel('x_1');
ylabel('x_2');
zlabel([Function_name,'( x_1 , x_2 )'])
grid off

%绘制收敛曲线
% CNT=20;   % 绘制图形随机选点的数量(可自行设置)
% k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
% iter=1:1:Max_iteration;
% subplot(1,2,2);
% plot(iter(k),cg_curve(k),'-*','LineWidth',1)
% hold on
% plot(iter(k),cg_curve1(k),'-p','LineWidth',1)
% legend('Improved','Original')
% grid off
% xlabel('Iteration')
% ylabel('Fitness')
% title('Iteration curve')
% axis tight
% grid on
% box on
% set(gca,'FontName','Times New Rome','FontSize',10);

% 差距不大时选择下面的绘图方法
CNT=20;   % 绘制图形随机选点的数量(可自行设置)
k=round(linspace(1,Max_iteration,CNT)); %随机选CNT个点
iter=1:1:Max_iteration;
subplot(1,2,2);
semilogy(iter(k),cg_curve(k),'-*','LineWidth',1)
hold on
semilogy(iter(k),cg_curve1(k),'-p','LineWidth',1)
legend('Improved','Original')
grid off
xlabel('Iteration')
ylabel('Fitness')
title('Iteration curve')
axis tight
grid on
box on
set(gca,'FontName','Times New Rome','FontSize',10);

%% 输出结果
disp('======改进后算法结果==========');
display(['最优方案 : ', num2str(Best_pos)]);
display(['最优值 : ', num2str(Best_score)]);
disp('======该进前算法结果============');
display(['最优方案 : ', num2str(Best_pos1)]);
display(['最优值 : ', num2str(Best_score1)]);





