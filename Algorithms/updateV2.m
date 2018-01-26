function [Vnew,beta2] = updateV2(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10,W21,W32,V,alpha,beta,t)
    [N,K] = size(U0);
    U = [U0;U1;U2;U3];
    I = sparse(eye(N));
    P = sparse([zeros(N,3*N) I]);
    Vstar = V - 2/(alpha*K)*((V*P*(U*U.')-y*U.')*P');
    Vnew = V + beta*(Vstar-V);
    if loss_fun(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10,W21,W32,V) <= loss_fun(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10,W21,W32,Vnew)
        beta2 = t*beta;
        Vnew = Vstar;
    else
        beta2 = min(beta/t,1);
    end
end
