function  erro=OBJ_lr(x,p_train,t_train)
% 超参数赋值

% 筛选变量
x_f=x(1:end);
p_train(:,x_f<0.5)=[];
if sum(sum(p_train))==0
    erro=100;
else
    %进行K折交叉验证
    rng(1)          % 为了重现性
    k=10;           % 10折交叉验证
    c = cvpartition(t_train,'KFold',k);
    Y=[];
    preY=[];
    for i=1:k
        trainingIndices = training(c,i); % 训练集索引
        valIndices = test(c,i); % 验证集索引
        XTrain = p_train(trainingIndices,:);
        YTrain = t_train(trainingIndices);
        Xval = p_train(valIndices,:);
        Yval = t_train(valIndices);
        % 训练逻辑回归
        mdl1 = fitglm(...
            XTrain,YTrain, ...
            'Distribution', 'binomial', ...
            'link', 'logit');
        [~,yfit]=predict(mdl1,Xval);
        Y=[Y;Yval];
        preY=[preY;yfit];
    end
    % 计算误差1-AUC
    [~,~,~,auc] = perfcurve(Y,preY(:,2),'1');
    erro=1-auc;
end
end