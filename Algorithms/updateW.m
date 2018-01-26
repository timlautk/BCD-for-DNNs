function [W10new,W21new,W32new,beta3] = updateW(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10,W21,W32,V,alpha,beta,t)
    [N,~] = size(U0);
    I = eye(N);    
    W10star = (gamma1*(U1*U0.')+alpha*W10)*pinv(gamma1*(U0*U0.')+alpha*I);
    W21star = (gamma2*(U2*U1.'-U0*U1.')+alpha*W21)*pinv(gamma2*(U1*U1.')+alpha*I);
    W32star = (-gamma3*(U0*U2.'+U1*U2.'-U3*U2.')+alpha*W32)*pinv(gamma3*(U2*U2.')+alpha*I);
    W10new = W10 + beta*(W10star-W10);
    W21new = W21 + beta*(W21star-W21);
    W32new = W32 + beta*(W32star-W32);
    if loss_fun(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10star,W21star,W32star,V) <= loss_fun(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10new,W21new,W32new,V)
        beta3 = t*beta;
        W10new = W10star;
        W21new = W21star;
        W32new = W32star;
    else
        beta3 = min(beta/t,1);
    end
end