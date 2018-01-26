function [W10new,W21new,W32new,beta3] = updateW2(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10,W21,W32,V,alpha,beta,t)
    W10star = W10 - gamma1/alpha*(-U1*U0.'+W10*(U0*U0.'));
    W21star = W21 - gamma2/alpha*(W21*(U1*U1.')-U2*U1.'+U0*U1.');
    W32star = W32 - gamma3/alpha*(W32*(U2*U2.')+U0*U2.'+U1*U2.'-U3*U2.');
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