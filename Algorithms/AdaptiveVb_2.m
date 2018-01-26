function [Vnew,b4new,beta1] = AdaptiveVb_2(gamma1,gamma2,gamma3,rho1,rho2,rho3,y,a0,a1,a2,a3,z1,z2,z3,W1,W2,W3,V,Vstar,b1,b2,b3,b4,b4star,beta,t,act)
    Vnew = V + beta*(Vstar-V);
    b4new = b4 + beta*(b4star-b4);
    if loss_fun_3(gamma1,gamma2,gamma3,rho1,rho2,rho3,y,a0,a1,a2,a3,z1,z2,z3,W1,W2,W3,Vstar,b1,b2,b3,b4star,act) <= loss_fun_3(gamma1,gamma2,gamma3,rho1,rho2,rho3,y,a0,a1,a2,a3,z1,z2,z3,W1,W2,W3,Vnew,b1,b2,b3,b4new,act)
        beta1 = t*beta;
        Vnew = Vstar;
        b4new = b4star;
    else
%         beta1 = min(beta/t,1);
        beta1 = beta;
    end

end