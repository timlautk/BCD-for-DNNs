function [a3new,beta2] = Adaptivea3(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,a3star,W1,W2,W3,V,b1,b2,b3,b4,beta,t)
    a3new = a3 + beta*(a3star-a3);
    if loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3star,W1,W2,W3,V,b1,b2,b3,b4) <= loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3new,W1,W2,W3,V,b1,b2,b3,b4)
        beta2 = t*beta;
        a3new = a3star;
    else
%         beta2 = min(beta/t,1);
        beta2 = beta;
    end
    
end