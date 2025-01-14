function [Tuna1_fit,Tuna1,Convergence_curve]=TSO(num1,num2,num3,Particles_no,Max_iter,Low,Up,Dim,fobj)

%% 运行群智能

Tuna1=zeros(1,Dim);   Tuna1_fit=inf;
%% 改进方式1：头部变异|混沌初始化
T=repmat(Low, Particles_no, 1)+chaos(num1,Particles_no,Dim) .* repmat((Up-Low), Particles_no, 1);
Iter=0;
aa=0.7;
z=0.05;
while Iter<Max_iter
    C=Iter/Max_iter;
    a1=aa+(1-aa)*C;
    a2=(1-aa)-(1-aa)*C;
    for i=1:size(T,1)

        Flag4ub=T(i,:)>Up;
        Flag4lb=T(i,:)<Low;
        T(i,:)=(T(i,:).*(~(Flag4ub+Flag4lb)))+Up.*Flag4ub+Low.*Flag4lb;

        fitness(i)=fobj(T(i,:));

        if fitness(i)<Tuna1_fit
            Tuna1_fit=fitness(i);  Tuna1=T(i,:);
        end
    end

    %% 改进方法3：尾部变异
    if num3 ~= 0 && Iter ~= 0
        for i=1:Particles_no
            T =integration(T,i,Tuna1,num3,t,Max_iter,Low,Up);
            Flag4ub=T(i,:)>Up;
            Flag4lb=T(i,:)<Low;
            T(i,:)=(T(i,:).*(~(Flag4ub+Flag4lb)))+Up.*Flag4ub+Low.*Flag4lb;
            fitness(i)=fobj(T(i,:));
            if fitness(i)<Tuna1_fit
                Tuna1_fit=fitness(i);  Tuna1=T(i,:);
            end
        end
    end

    %---------------- Memory saving-------------------
    if Iter==0
        fit_old=fitness;  C_old=T;
    end

    for i=1:Particles_no
        if fit_old(i)<fitness(i)
            fitness(i)=fit_old(i); T(i,:)=C_old(i,:);
        end
    end

    C_old=T;  fit_old=fitness;
    %-------------------------------------------------

    t=(1-Iter/Max_iter)^(Iter/Max_iter);


    if rand<z
        T(1,:)= (Up-Low)*rand+Low;
    else
        if  0.5<rand
            r1=rand;
            Beta=exp(r1*exp(3*cos(pi*((Max_iter-Iter+1)/Max_iter))))*(cos(2*pi*r1));
            if  C>rand
                T(1,:)=a1.*(Tuna1+Beta*abs(Tuna1-T(1,:)))+a2.*T(1,:); %Equation (8.3)

            else
                IndivRand=rand(1,Dim).*(Up-Low)+Low;
                T(1,:)=a1.*(IndivRand+Beta*abs(IndivRand-T(i,:)))+a2.*T(1,:);%Equation (8.1)
            end
        else
            TF = (rand>0.5)*2-1;
            if 0.5>rand
                T(1,:)=Tuna1+rand(1,Dim).*(Tuna1-T(1,:))+TF.*t^2.*(Tuna1-T(1,:));%Equation (9.1)
            else
                T(1,:) =TF.* t^2.*T(1,:);%Equation (9.2)
            end

        end

    end
    %% 改进方法2：融合改进
    for i=2:Particles_no
        if rand > 1-sqrt(Iter/Max_iter) && num2 ~= 0  % 满足条件执行变异
            T =integration(T,i,Tuna1,num2,t,Max_iter,Low,Up);
        else

            if rand<z

                T(i,:)= (Up-Low)*rand+Low;
            else
                if  0.5<rand
                    r1=rand;
                    Beta=exp(r1*exp(3*cos(pi*((Max_iter-Iter+1)/Max_iter))))*(cos(2*pi*r1));
                    if  C>rand
                        T(i,:)=a1.*(Tuna1+Beta*abs(Tuna1-T(i,:)))+a2.*T(i-1,:);%Equation (8.4)
                    else

                        IndivRand=rand(1,Dim).*(Up-Low)+Low;
                        T(i,:)=a1.*(IndivRand+Beta*abs(IndivRand-T(i,:)))+a2.*T(i-1,:);%Equation (8.2)
                    end
                else
                    TF = (rand>0.5)*2-1;
                    if 0.5>rand
                        T(i,:)=Tuna1+rand(1,Dim).*(Tuna1-T(i,:))+TF*t^2.*(Tuna1-T(i,:)); %Equation (9.1)
                    else
                        T(i,:) = TF*t^2.*T(i,:);%Equation (9.2)
                    end
                end
            end
        end
    end
    
 

    Iter=Iter+1;
    Convergence_curve(Iter)=Tuna1_fit;

end

end

