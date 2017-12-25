function [best,fmin]=BBA(n, A, r, d, Max_iter,train,train_labels)

Qmin=0;         % Frequency minimum
Qmax=2;         % Frequency maximum
N_iter=0;       % Total number of function evaluations
Q=zeros(n,1);   % Frequency
v=zeros(n,d);   % Velocities
Sol=zeros(n,d);
for i=1:n,
    for j=1:d % For dimension
        if rand<=0.5
            Sol(i,j)=0;
        else
            Sol(i,j)=1;
        end
    end
end
Sol(1,:)=ones(1,d);
for i=1:n
    Fitness(i)=MyCost(Sol(i,:),train,train_labels);
end
% Find the current best
[fmin,I]=min(Fitness);
best=Sol(I,:);

while (N_iter<Max_iter)
        N_iter=N_iter+1;
        for i=1:n,
            for j=1:d
                Q(i)=Qmin+(Qmin-Qmax)*rand; 
                v(i,j)=v(i,j)+(Sol(i,j)-best(j))*Q(i); 

                V_shaped_transfer_function=abs((2/pi)*atan((pi/2)*v(i,j))); 
                
                if rand<V_shaped_transfer_function 
                    Sol(i,j)=~Sol(i,j);
                else
                    Sol(i,j)=Sol(i,j);
                end
                
                if rand>r  % Pulse rate
                      Sol(i,j)=best(j);
                end   
               
            end       
            
           Fnew=MyCost(Sol(i,:),train,train_labels); 
     
           if (Fnew<=Fitness(i)) && (rand<A)  
                Sol(i,:)=Sol(i,:);
                Fitness(i)=Fnew;
           end

          % Update the current best
          if Fnew<=fmin,
                best=Sol(i,:);
                fmin=Fnew;
          end
        end
        
     
end


