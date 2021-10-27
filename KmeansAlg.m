clc
clear all
close all
K=2;% number of clusters
N=20000;
%GMM parameters:
c1=0.5;
c2=0.5;

mu1 = [1 1];          % Mean of the 1st component
sigma1 = [1 0; 0 2]; % Covariance of the 1st component
mu2 = [3 3];        % Mean of the 2nd component
sigma2 = [2 0; 0 0.5];  % Covariance of the 2nd component
%%%% GMM 
r1 = mvnrnd(mu1,sigma1,N*c1);
r2 = mvnrnd(mu2,sigma2,N*c2);

X = [r1; r2];
%  scatter(X(:,1),X(:,2));
% z1=X(1,:);
% z2=X(1500,:);
z1=rand(1,2)*5;
z2=rand(1,2)*5;
t=[ones(N*c1,1); 2*ones(N*c2,1)];
z=[z1;z2];
%cluster dataset using nearest neightbor rule:
for iteration=1:12
    for i=1:N
        for j=1:K
            distance(i,j)=sqrt(sum((X(i,:) - z(j,:)) .^ 2));
        end
    end
    [argvalue, argmin] = min(distance');

    for i=1:K
       z(i,:)= mean(X(argmin==i,:));

    end
%    figure(1)
% scatter(X(:,1),X(:,2),[],argmin,'.');

errorRate = sum(t'~=argmin,2)/N;
errorRate = min(errorRate,1-errorRate);
figure(1)
clf
hold on
scatter(X(:,1),X(:,2),60,argmin,'.');
scatter(z(1,1),z(1,2),40,'r','filled');
scatter(z(2,1),z(2,2),40,'r','filled');
title('K - Means algorithm');
hold off
end
errorRate
%%%%%%%GMM EM alg%%%%%%%
c1_estimated=rand(1);
c2_estimated=1-c1_estimated;
mu1_estimated = rand(1,2)*5;          
sigma1_estimated = eye(2).*rand(2)*5;
mu2_estimated = rand(1,2)*5;       
sigma2_estimated =  eye(2).*rand(2)*5;
for i=1:15
    %E step
    alj=calculate_alj(N,X,c1_estimated,c2_estimated,sigma1_estimated,sigma2_estimated,mu1_estimated,mu2_estimated);
    %M step
    al=sum(alj,1);
    c1_estimated=al(1)/N;
    c2_estimated=al(2)/N;
    mu1_estimated=sum(alj(:,1).*X,1)/al(1);
    mu2_estimated=sum(alj(:,2).*X,1)/al(2);
    sigma1_estimated=(1/al(1))*diag(sum(alj(:,1).*((X-mu1_estimated).^2),1));
    sigma2_estimated=(1/al(2))*diag(sum(alj(:,2).*((X-mu2_estimated).^2),1));
    [~ ,class]= max(alj');
    
   

%plot results
errorRate=sum(t'~=class,2)/N;
errorRate=min(errorRate,1-errorRate);
figure(2)
clf
hold on
scatter(X(:,1),X(:,2),60,class,'.');
scatter(mu1_estimated(1,1),mu1_estimated(1,2),40,'r','filled');
scatter(mu2_estimated(1,1),mu2_estimated(1,2),40,'r','filled');
title('EM algorithm');

mu = [mu1_estimated;mu2_estimated];
sigma = cat(3,diag(sigma1_estimated)',diag(sigma2_estimated)'); % 1-by-2-by-2 array
gm = gmdistribution(mu,sigma);
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm,[x0 y0]),x,y);
fcontour(gmPDF,[-8 6])
hold off
end
errorRate=min(errorRate,1-errorRate)

function p = calculate_p(x,sigma,mu)
    p=(1/sqrt(2*pi))^2 *(det(sigma))^(-0.5) * exp(-0.5*(x-mu)*inv(sigma)*(x-mu)'); 
end
function alj=calculate_alj(N,X,c1_estimated,c2_estimated,sigma1_estimated,sigma2_estimated,mu1_estimated,mu2_estimated)
for i=1:N
    x=X(i,:);
    one = c1_estimated * calculate_p(x,sigma1_estimated,mu1_estimated);
    two= c2_estimated * calculate_p(x,sigma2_estimated,mu2_estimated);
    alj(i,1)=one/(one+two);
    alj(i,2)=two/(one+two);
end

end

 
