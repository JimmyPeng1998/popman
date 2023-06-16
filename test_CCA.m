clear
clc
clf

% This test is for the comparison of Riemannian gradient descent and
% Riemannian conjugate gradient methods under different metrics for solving
% the canonical correlation analysis problem.



%% Default settings
% Problem statements
dx=200;
dy=100;
n=3000;
m=3;
N=sparse(1:m,1:m,m:-1:1);

lambda_x=1e-6;
lambda_y=1e-6;

X=rand(n,dx);
Y=rand(n,dy);
Sigma_xx=X'*X+lambda_x*speye(dx);
Sigma_yy=Y'*Y+lambda_y*speye(dy);
Sigma_xy=X'*Y;

% Compute the ground truth
[U,LambdaX]=eig(Sigma_xx);
[V,LambdaY]=eig(Sigma_yy);
[Utrue,S,Vtrue]=svd((U/sqrt(LambdaX))*U'*Sigma_xy*(V/sqrt(LambdaY))*V');
s=diag(S);
truth=-sum(s(1:m)'.*(m:-1:1));
xtrue.U=((U*sqrt(LambdaX))*U')\Utrue(:,1:m);
xtrue.V=((V*sqrt(LambdaY))*V')\Vtrue(:,1:m);

symm = @(X) .5*(X+X');


% Initial guess
x0=struct;
x0.U=gqf(randn(dx,m),Sigma_xx);
x0.V=gqf(randn(dy,m),Sigma_yy);


% Options
solver='RCG';
% solver='RGD';
% condition_number=true; % Computing the condition number is expensive.
condition_number=false;
compared_solvers={[solver '(E)'];[solver '(L1)'];[solver '(L2)'];[solver '(L12)'];[solver '(LR12)']};
compared_metrics=[1 2 3 4 5]; 

options.maxiter=1000;
options.statsfun=@(problem, x, stats) myStatsFun(problem, x, stats, xtrue.U, xtrue.V, m);
% options.verbosity=0; % Cancel the output



%% Generate preconditioners
B1=cell(5,1);
B2=cell(5,1);
info=cell(5,1);

% Euclidean metric (E)
B1{1}.left=@(X) speye(dx); B1{1}.right=@(X) speye(m);
B2{1}.left=@(X) speye(dy); B2{1}.right=@(X) speye(m);
B1{1}.leftdot=@(X,eta) sparse(dx); B1{1}.rightdot=@(X,eta) sparse(m);
B2{1}.leftdot=@(X,eta) sparse(dy); B2{1}.rightdot=@(X,eta) sparse(m);

% Metric (L1)
B1{2}.left=@(X) Sigma_xx;  B1{2}.right=@(X) speye(m);
B2{2}.left=@(X) speye(dy); B2{2}.right=@(X) speye(m);
B1{2}.leftdot=@(X,eta) sparse(dx); B1{2}.rightdot=@(X,eta) sparse(m);
B2{2}.leftdot=@(X,eta) sparse(dy); B2{2}.rightdot=@(X,eta) sparse(m);

% Metric (L2)
B1{3}.left=@(X) speye(dx); B1{3}.right=@(X) speye(m);
B2{3}.left=@(X) Sigma_yy;  B2{3}.right=@(X) speye(m);
B1{3}.leftdot=@(X,eta) sparse(dx); B1{3}.rightdot=@(X,eta) sparse(m);
B2{3}.leftdot=@(X,eta) sparse(dy); B2{3}.rightdot=@(X,eta) sparse(m);

% Metric (L12)
B1{4}.left=@(X) Sigma_xx;  B1{4}.right=@(X) speye(m);
B2{4}.left=@(X) Sigma_yy;  B2{4}.right=@(X) speye(m);
B1{4}.leftdot=@(X,eta) sparse(dx); B1{4}.rightdot=@(X,eta) sparse(m);
B2{4}.leftdot=@(X,eta) sparse(dy); B2{4}.rightdot=@(X,eta) sparse(m);

% Metric (LR12)
B1{5}.left=@(X) Sigma_xx;  B1{5}.right=@(X) rightmetric(X,Sigma_xy,m,N);
B2{5}.left=@(X) Sigma_yy;  B2{5}.right=@(X) rightmetric(X,Sigma_xy,m,N);
B1{5}.leftdot=@(X,eta) sparse(dx); B1{5}.rightdot=@(X,eta) 0.5*((eta.U'*Sigma_xy*X.V*N+X.U'*Sigma_xy*eta.V*N)+(eta.U'*Sigma_xy*X.V*N+X.U'*Sigma_xy*eta.V*N)');
B2{5}.leftdot=@(X,eta) sparse(dy); B2{5}.rightdot=@(X,eta) 0.5*((eta.V'*Sigma_xy'*X.U*N+X.V'*Sigma_xy'*eta.U*N)+(eta.V'*Sigma_xy'*X.U*N+X.V'*Sigma_xy'*eta.U*N)');




for i=compared_metrics
    % Problem
    M=ProdStiefelGeneralFactory_Precon(dx,dy,m,Sigma_xx,Sigma_yy,B1{i},B2{i});
    problem.M=M;
    problem.cost=@(X) Cost(X.U,Sigma_xy,X.V,N);
    problem.egrad=@(X) struct('U', -Sigma_xy*X.V*N, 'V', -Sigma_xy'*X.U*N);
    problem.ehess=@(X,eta) struct('U', -Sigma_xy*eta.V*N, 'V', -Sigma_xy'*eta.U*N);
    
    % Method (Automatically check the availability of Manopt)
    try
        if strcmp(solver,'RGD')==1
            [x, ~, info{i}, ~]=steepestdescent(problem,x0,options);
        else
            [x, ~, info{i}, ~]=conjugategradient(problem,x0,options);
        end
    catch
        error('Please install <a href="https://www.manopt.org/">Manopt</a> first')
    end
    
    % Condition number at a local minimizer
    if condition_number==true
        % Note that computing the condition number is expensive. 
        lambda = hessianspectrum(problem, xtrue); 
        lambda = lambda(end:-1:1);
        fprintf('The condition number: %6.2e\n',real(lambda(1))/real(lambda(end)))
        
    end
end




%% Plotting results
lwidth=4;
msize=8;
colors=get(gca,'ColorOrder');
Markers={'','-^','-^','','-^','-^','-^',':s',':o',':d'};

% iter vs residual
for i=compared_metrics
    semilogy(0:size([info{i}.cost],2)-1,[info{i}.cost]-truth,'LineWidth',lwidth,'MarkerSize',msize,'MarkerFaceColor',colors(i,:),'MarkerEdgeColor','none','Color',colors(i,:))
    hold on
end

title([solver ' for CCA'])
xlabel('#iter')
ylabel('$f(\mathbf{U},\mathbf{V})-f_{\min}$','Interpreter','latex')
set(gca,'FontSize',16)
if strcmp(solver,'RGD')==1
    legend(compared_solvers{compared_metrics},'Location','southwest')
else
    legend(compared_solvers{compared_metrics})
end


% time (s) vs residual
figure()
for i=compared_metrics
    semilogy([info{i}.time],[info{i}.cost]-truth,'LineWidth',lwidth,'MarkerSize',msize,'MarkerFaceColor',colors(i,:),'MarkerEdgeColor','none','Color',colors(i,:))
    hold on
end

title([solver ' for CCA'])
xlabel('time (s)')
ylabel('$f(\mathbf{U},\mathbf{V})-f_{\min}$','Interpreter','latex')
set(gca,'FontSize',16)
if strcmp(solver,'RGD')==1
    legend(compared_solvers{compared_metrics},'Location','southwest')
else
    legend(compared_solvers{compared_metrics})
end


% Generalized QR decomposition
function X = gqf(Y,B)
    % From Manopt
    R = chol(Y'*(B*Y));
    X = Y / R;
end

% Compute the cost function
function tr=Cost(U,A,V,N)
    temp1=(U'*A)';
    temp2=V*N;
    tr=-temp1(:)'*temp2(:);
end

% Construct the right preconditioners
function M=rightmetric(X,Sigma_xy,m,N)
    temp=X.U'*Sigma_xy*X.V*N;
    M=0.5*(temp+temp');
    [U,V]=eig(M);
    M=U*sqrt(sparse(V).^2+1e-15*speye(m))*U';
end

% Compute the subspace distances
function stats=myStatsFun(problem, x, stats, Utrue, Vtrue, p)
    stats.normU=norm(x.U*x.U'-Utrue*Utrue','fro');
    stats.normV=norm(x.V*x.V'-Vtrue*Vtrue','fro');
end
