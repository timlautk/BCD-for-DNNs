function [W3new,b3new,beta3] = AdaptiveWb3(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2,W3,V,W3star,b1,b2,b3,b4,b3star,beta,t)
    W3new = W3 + beta*(W3star-W3);
    b3new = b3 + beta*(b3star-b3);
    if loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2,W3star,V,b1,b2,b3star,b4) <= loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2,W3new,V,b1,b2,b3new,b4)
        beta3 = t*beta;
        W3new = W3star;
        b3new = b3star;
    else
%         beta3 = min(beta/t,1);
        beta3 = beta;
    end
    
end