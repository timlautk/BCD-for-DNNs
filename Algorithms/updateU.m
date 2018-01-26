function [U1new,U2new,U3new,beta1] = updateU(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10,W21,W32,V,alpha,beta,t)
    [N,K] = size(U0);
    U = [U0;U1;U2;U3];
    I = sparse(eye(N));
    Q = 2*[
        gamma1/2*(W10'*W10)+(gamma2+gamma3)/2*I, gamma2*W21+gamma3/2*I, -gamma2/2*I+gamma3*W32, -gamma3/2*I;
        -gamma1*W10+gamma3/2*I, (gamma1+gamma3)/2*I+gamma2/2*(W21'*W21), gamma3*W32, -gamma3/2*I;
        -gamma2/2*I, -gamma2*W21, gamma2/2*I+gamma3/2*(W32'*W32), zeros(N);
        -gamma3/2*I, -gamma3/2*I, -gamma3*W32, gamma3/2*I
        ];
    P = sparse([zeros(N,3*N) I]);
    Ustar = (2/K*(V*P)'*(V*P)+Q+alpha*eye(4*N))\(2/K*(V*P)'*y+alpha*U);
    Ustar = max(0,Ustar);
%     Ustar = max(-1,min(Ustar,1));
    U = U + beta*(Ustar-U);
    U1new = U(N+1:2*N,:);
    U2new = U(2*N+1:3*N,:);
    U3new = U(3*N+1:end,:);
    if loss_fun(gamma1,gamma2,gamma3,y,U0,Ustar(N+1:2*N,:),Ustar(2*N+1:3*N,:),Ustar(3*N+1:end,:),W10,W21,W32,V) <= loss_fun(gamma1,gamma2,gamma3,y,U0,U1new,U2new,U3new,W10,W21,W32,V)
        beta1 = t*beta;
        U1new = Ustar(N+1:2*N,:);
        U2new = Ustar(2*N+1:3*N,:);
        U3new = Ustar(3*N+1:end,:);
    else
        beta1 = min(beta/t,1);
    end
end