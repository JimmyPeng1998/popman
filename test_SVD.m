clear
clc
clf

% This test is for the comparison of Riemannian gradient descent and
% Riemannian conjugate gradient methods under different metrics for solving
% the truncated singular value decomposition problem.



%% Default settings
% Problem statements
m=1000;
n=500;
p=10;


[Utrue,~]=qr(randn(m));
[Vtrue,~]=qr(randn(n));
base=1.5;
s=max(base.^(0:-1:-n+1),eps);
N=sparse(1:p,1:p,p:-1:1);
s=sort(s,'descend');
A=Utrue(:,1:n)*diag(s)*Vtrue';

% Compute the ground truth
truth=-s(1:p)*diag(N);
xtrue.U=Utrue(:,1:p);
xtrue.V=Vtrue(:,1:p);

symm = @(X) .5*(X+X');


% Initial guess
x0=struct;
[x0.U,~]=qr(randn(m,p));
[x0.V,~]=qr(randn(n,p));
x0.U=x0.U(:,1:p);
x0.V=x0.V(:,1:p);


% Options
solver='RCG';
% solver='RGD';
% condition_number=true; % Computing the condition number is expensive.
condition_number=false;
compared_solvers={[solver '(E)'];[solver '(R12)']};
compared_metrics=[1 2]; 

options.maxiter=1000;
options.statsfun=@(problem, x, stats) myStatsFun(problem, x, stats, xtrue.U, xtrue.V, p);
% options.verbosity=0; % Cancel the output



%% Generate preconditioners
B1=cell(2,1);
B2=cell(2,1);
info=cell(2,1);

% Euclidean metric (E)
B1{1}.left=@(X) speye(m); B1{1}.right=@(X) speye(p);
B2{1}.left=@(X) speye(n); B2{1}.right=@(X) speye(p);
B1{1}.leftdot=@(X,eta) sparse(m); B1{1}.rightdot=@(X,eta) sparse(p);
B2{1}.leftdot=@(X,eta) sparse(n); B2{1}.rightdot=@(X,eta) sparse(p);


% Metric (R12)
B1{2}.left=@(X) speye(m);  B1{2}.right=@(X) rightmetric(X,A,p,N);
B2{2}.left=@(X) speye(n);  B2{2}.right=@(X) rightmetric(X,A,p,N);
B1{2}.leftdot=@(X,eta) sparse(m); B1{2}.rightdot=@(X,eta) 0.5*((eta.U'*A*X.V*N+X.U'*A*eta.V*N)+(eta.U'*A*X.V*N+X.U'*A*eta.V*N)');
B2{2}.leftdot=@(X,eta) sparse(n); B2{2}.rightdot=@(X,eta) 0.5*((eta.V'*A'*X.U*N+X.V'*A'*eta.U*N)+(eta.V'*A'*X.U*N+X.V'*A'*eta.U*N)');




for i=compared_metrics
    % Problem
    M=ProdStiefelGeneralFactory_Precon(m,n,p,speye(m),speye(n),B1{i},B2{i});
    problem.M=M;
    problem.cost=@(X) Cost(X.U,A,X.V,N);
    problem.egrad=@(X) struct('U', -A*X.V*N, 'V', -A'*X.U*N);
    problem.ehess=@(X,eta) struct('U', -A*eta.V*N, 'V', -A'*eta.U*N);
    
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
semilogy(0:size([info{i}.cost],2)-1,[info{i}.cost]-truth,'LineWidth',lwidth,'MarkerSize',msize,'MarkerFaceColor',colors(i,:),'MarkerEdgeColor','none','Color',colors(1+(i-1)*4,:))
hold on
end

title([solver ' for SVD'])
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
semilogy([info{i}.time],[info{i}.cost]-truth,'LineWidth',lwidth,'MarkerSize',msize,'MarkerFaceColor',colors(i,:),'MarkerEdgeColor','none','Color',colors(1+(i-1)*4,:))
hold on
end

title([solver ' for SVD'])
xlabel('time (s)')
ylabel('$f(\mathbf{U},\mathbf{V})-f_{\min}$','Interpreter','latex')
set(gca,'FontSize',16)
if strcmp(solver,'RGD')==1
    legend(compared_solvers{compared_metrics},'Location','southwest')
else
    legend(compared_solvers{compared_metrics})
end






% Compute the cost function
function tr=Cost(U,A,V,N)
    temp1=(U'*A)';
    temp2=V*N;
    tr=-temp1(:)'*temp2(:);
end

% Construct the right preconditioners
function M=rightmetric(X,A,p,N)
    temp=X.U'*A*(X.V*N);
    M=0.5*(temp+temp');
    [U,V]=eig(M);
    M=U*sqrt(sparse(V).^2+1e-15*speye(p))*U';
end

% Compute the subspace distances
function stats=myStatsFun(problem, x, stats, Utrue, Vtrue, p)
    stats.normU=norm(x.U*x.U'-Utrue*Utrue','fro');
    stats.normV=norm(x.V*x.V'-Vtrue*Vtrue','fro');
end
