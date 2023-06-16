function M = ProdStiefelGeneralFactory_Precon(m,n,p,Sigma_xx,Sigma_yy,B1,B2)
% function M = ProdStiefelGeneralFactory_Precon(m,n,p,Sigma_xx,Sigma_yy,B1,B2)
% Providing a product manifold of two generalized Stiefel manifold
% St(p,m,Sigma_xx) x St(p,n,Sigma_yy) endowed with a preconditioned metric
% g_X(xi,eta)=tr(xi'*B1.left(X)*eta*B1.right(X))+tr(xi'*B2.left(X)*eta*B2.right(X)),
% where X=(X.U,X.V)
%
% To access the Riemannian Hessian at a local minimizer, please
% simultaneously provide the derivative of all preconditioners along eta
% and store it as
% B1.leftdot, B1.rightdot, B2.leftdot, B2.rightdot
%
% Original author: Renfeng Peng, Apr. 05, 2023.


    M.name=@() sprintf('Product manifold of two generalized Stiefel manifold St(%d,%d,Sigma_xx) x St(%d,%d,Sigma_yy).',...
        p,m,p,n);

    M.dim=@() (m+n)*p-p*(p+1);



    % Preparing the precomputations
    function X=prepare_precon(X)
        % Precomputations for metric
        if ~all(isfield(X,{'B11','B12','B21','B22'})==1)
            X.B11=B1.left(X);
            X.B12=B1.right(X);
            X.B21=B2.left(X);
            X.B22=B2.right(X);
        end
        
        
        % Precomputations for egrad2rgrad
        if ~all(isfield(X,{'Ut_Sigmaxx,Ut_Sigmaxx_invB11,Ut_Sigmaxx_invB11_Sigmaxx_U',...
                'Vt_Sigmayy','Vt_Sigmayy_invB21','Vt_Sigmayy_invB21_Sigmayy_V'})==1)
            X.Ut_Sigmaxx=X.U'*Sigma_xx;
            if isequal(X.B11,Sigma_xx)
                X.Ut_Sigmaxx_invB11=X.U';
                X.Ut_Sigmaxx_invB11_Sigmaxx_U=speye(p);
            else
                X.Ut_Sigmaxx_invB11=X.Ut_Sigmaxx/X.B11;
                X.Ut_Sigmaxx_invB11_Sigmaxx_U=X.Ut_Sigmaxx_invB11*X.Ut_Sigmaxx';
            end
            
            
            X.Vt_Sigmayy=X.V'*Sigma_yy;
            if isequal(X.B21,Sigma_yy)
                X.Vt_Sigmayy_invB21=X.V';
                X.Vt_Sigmayy_invB21_Sigmayy_V=speye(p);
            else
                X.Vt_Sigmayy_invB21=X.Vt_Sigmayy/X.B21;
                X.Vt_Sigmayy_invB21_Sigmayy_V=X.Vt_Sigmayy_invB21*X.Vt_Sigmayy';
            end
        end
    end

    M.inner=@preconInner;
    function ip=preconInner(X,xi,eta)
        X=prepare_precon(X);
        temp1=X.B11*xi.U*X.B12;
        temp2=X.B21*xi.V*X.B22;
        ip=temp1(:)'*eta.U(:)+temp2(:)'*eta.V(:);
    end


    M.norm = @(X, eta) sqrt(M.inner(X, eta, eta));

    M.dist = @(x, y) error('ProdStiefelGeneralFactory_Precon.dist not implemented yet.');

    M.typicaldist = @(x, y) error('ProdStiefelGeneralFactory_Precon.typicaldist not implemented yet.');

    % Compute the Riemannian gradient from the Euclidean gradient
    M.egrad2rgrad=@egrad2rgrad;
    function rgrad=egrad2rgrad(X,egrad)
        % Preparation
        X=prepare_precon(X);
        
        % Solving Lyapunov equations
        LHS1=X.B12*X.Ut_Sigmaxx_invB11_Sigmaxx_U;
        temp=X.B12*X.Ut_Sigmaxx_invB11*egrad.U;
        RHS1=temp+temp';
        S1=lyap(LHS1,-RHS1);
        
        LHS2=X.B22*X.Vt_Sigmayy_invB21_Sigmayy_V;
        temp=X.B22*X.Vt_Sigmayy_invB21*egrad.V;
        RHS2=temp+temp';
        S2=lyap(LHS2,-RHS2);
        
        % Compute the Riemannian gradients
        rgrad.U=(X.B11\egrad.U-X.Ut_Sigmaxx_invB11'*S1)/X.B12;
        rgrad.V=(X.B21\egrad.V-X.Vt_Sigmayy_invB21'*S2)/X.B22;
        
    end

    symm = @(X) .5*(X+X');
    % Compute the Riemannian Hessian from the Euclidean Hessian
    M.ehess2rhess = @ehess2rhess;
    function Hess = ehess2rhess(X, egrad, ehess, eta)
        warning('manopt:ProdStiefelGeneralFactory_Precon:ehess2rhess', ...
            'The ehess2rhess is only applicable for the Riemannian Hessian at a critical point.');
        X=prepare_precon(X);
        
        % Check all nessary fields
        try
            if ~all(isfield(B1,{'leftdot,rightdot'})==1) || ~all(isfield(B2,{'leftdot,rightdot'})==1)
                X.B11dot=B1.leftdot(X,eta);
                X.B12dot=B1.rightdot(X,eta);
                X.B21dot=B2.leftdot(X,eta);
                X.B22dot=B2.rightdot(X,eta);
            end
        catch
            error('Please provide the derivatives of preconditioners along eta.');
        end
        
        % Compute the Riemannian gradient
        LHS1=X.B12*X.Ut_Sigmaxx_invB11_Sigmaxx_U;
        temp=X.B12*X.Ut_Sigmaxx_invB11*egrad.U;
        RHS1=temp+temp';
        S1=lyap(LHS1,-RHS1);
        
        LHS2=X.B22*X.Vt_Sigmayy_invB21_Sigmayy_V;
        temp=X.B22*X.Vt_Sigmayy_invB21*egrad.V;
        RHS2=temp+temp';
        S2=lyap(LHS2,-RHS2);
        
        rgrad.U=(X.B11\egrad.U-X.Ut_Sigmaxx_invB11'*S1)/X.B12;
        rgrad.V=(X.B21\egrad.V-X.Vt_Sigmayy_invB21'*S2)/X.B22;
        
        
        
        % Compute the Riemannian Hessian
        temp=X.B12dot*X.Ut_Sigmaxx_invB11*egrad.U+...
            X.B12*eta.U'*Sigma_xx*(X.B11\egrad.U)-...
            X.B12*X.Ut_Sigmaxx_invB11*X.B11dot*(X.B11\egrad.U)+...
            X.B12*X.Ut_Sigmaxx_invB11*ehess.U; % Right hand terms
        temp=temp-(X.B12dot*X.Ut_Sigmaxx_invB11_Sigmaxx_U*S1+...
            X.B12*eta.U'*Sigma_xx*(X.B11\(Sigma_xx*X.U))*S1-...
            X.B12*X.U'*Sigma_xx*(X.B11\(X.B11dot*(X.B11\(Sigma_xx*X.U))))*S1+...
            X.B12*X.U'*Sigma_xx*(X.B11\(Sigma_xx*eta.U))*S1);
        RHS1=temp+temp';
        S1dot=lyap(LHS1,-RHS1);
        
        temp=X.B22dot*X.Vt_Sigmayy_invB21*egrad.V+...
            X.B22*eta.V'*Sigma_yy*(X.B21\egrad.V)-...
            X.B22*X.Vt_Sigmayy_invB21*X.B21dot*(X.B21\egrad.V)+...
            X.B22*X.Vt_Sigmayy_invB21*ehess.V; % Right hand terms
        temp=temp-(X.B22dot*X.Vt_Sigmayy_invB21_Sigmayy_V*S2+...
            X.B22*eta.V'*Sigma_yy*(X.B21\(Sigma_yy*X.V))*S2-...
            X.B22*X.V'*Sigma_yy*(X.B21\(X.B21dot*(X.B21\(Sigma_yy*X.V))))*S2+...
            X.B22*X.V'*Sigma_yy*(X.B21\(Sigma_yy*eta.V))*S2);
        RHS2=temp+temp';
        S2dot=lyap(LHS2,-RHS2);
        
        
        Hess.U=-X.B11\(X.B11dot*rgrad.U)+(X.B11\ehess.U)/X.B12-rgrad.U*X.B12dot*X.B12-...
            (X.B11\(Sigma_xx*(eta.U*S1+X.U*S1dot)))/X.B12;
        
        Hess.V=-X.B21\(X.B21dot*rgrad.V)+(X.B21\ehess.V)/X.B22-rgrad.V*X.B22dot*X.B22-...
            (X.B21\(Sigma_yy*(eta.V*S2+X.V*S2dot)))/X.B22;
        
        
        Hess=projection(X,Hess);
    end


    % Compute the projection with respect to the preconditioned metric to the
    % tangent space
    M.proj = @projection;
    function Proj = projection(X, eta)
        X=prepare_precon(X);
        
        LHS1=X.B12*X.Ut_Sigmaxx_invB11_Sigmaxx_U;
        temp=X.B12*(X.Ut_Sigmaxx_invB11*eta.U)*X.B12;
        RHS1=temp+temp';
        S1=lyap(LHS1,-RHS1);
        
        LHS2=X.B22*X.Vt_Sigmayy_invB21_Sigmayy_V;
        temp=X.B22*(X.Vt_Sigmayy_invB21*eta.V)*X.B22;
        RHS2=temp+temp';
        S2=lyap(LHS2,-RHS2);
        
        
        Proj.U=eta.U-(X.Ut_Sigmaxx_invB11'*S1)/X.B12;
        Proj.V=eta.V-(X.Vt_Sigmayy_invB21'*S2)/X.B22;
    end


    M.tangent = M.proj;
    M.tangent2ambient = @(X, eta) eta;



    M.exp = @exponential;
    function Y = exponential(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        Y = qr_retraction(X, eta, t);
        warning('manopt:ProdStiefelGeneralFactory_Precon:exp', ...
            ['Not implemented yet. Used retraction instead.']);
    end




    % Implementing generalized QR retraction
    M.retr = @qr_retraction;
    function Y = qr_retraction(X, eta, t)
        if nargin < 3
            t = 1.0;
        end
        
        Y.U = gqf(X.U+t*eta.U,Sigma_xx);
        Y.V = gqf(X.V+t*eta.V,Sigma_yy);
        
        
        Y = prepare_precon(Y);
    end



    M.hash = @(X) ['z' hashmd5([sum(X.U(:)) ; sum(X.V(:))])]; % Efficient, suggested by Bart Vandereycken.

    M.rand = @random;
    function X = random()
        X.U=gqf(randn(m,p),Sigma_xx);
        X.V=gqf(randn(n,p),Sigma_yy);
    end

    M.randvec = @randomvec;
    function eta = randomvec(X)
        % A random vector on the tangent space
        eta.U = randn(m, p);
        eta.V = randn(n, p);
        eta = projection(X, eta);
        nrm = M.norm(X, eta);
        eta.U = eta.U / nrm;
        eta.V = eta.V / nrm;
    end

    M.lincomb = @lincomb;
    function d = lincomb(X, a1, d1, a2, d2) %#ok<INUSL>
        
        if nargin == 3
            d.U = a1*d1.U;
            d.V = a1*d1.V;
        elseif nargin == 5
            d.U = a1*d1.U + a2*d2.U;
            d.V = a1*d1.V + a2*d2.V;
        else
            error('Bad use of ProdStiefelGeneralFactory_Precon.lincomb.');
        end
        
    end





    M.zerovec = @(X) struct('U', zeros(m, p), 'V', zeros(n, p));

    M.transp = @(x1, x2, d) projection(x2, d);

    M.vec = @(X, u_mat) [u_mat.U(:); u_mat.V(:)];
    M.mat = @(X, u_vec) struct('U', reshape(u_vec(1:m*p), [m, p]), 'V', reshape(u_vec(1+m*p:end), [n, p]));
    M.vecmatareisometries = @() false;



    function X = gqf(Y,B)
        % From Manopt
        R = chol(Y'*(B*Y));
        X = Y / R;
    end
end
