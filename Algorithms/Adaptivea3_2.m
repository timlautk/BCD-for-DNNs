function [a3new,beta2] = Adaptivea3_2(gamma1,gamma2,gamma3,rho1,rho2,rho3,y,a0,a1,a2,a3,a3star,z1,z2,z3,W1,W2,W3,V,b1,b2,b3,b4,beta,t,act)
    a3new = a3 + beta*(a3star-a3);
    if loss_fun_3(gamma1,gamma2,gamma3,rho1,rho2,rho3,y,a0,a1,a2,a3star,z1,z2,z3,W1,W2,W3,V,b1,b2,b3,b4,act) <= loss_fun_3(gamma1,gamma2,gamma3,rho1,rho2,rho3,y,a0,a1,a2,a3new,z1,z2,z3,W1,W2,W3,V,b1,b2,b3,b4,act)
        beta2 = t*beta;
        a3new = a3star;
    else
%         beta2 = min(beta/t,1);
        beta2 = beta;
    end
    
end