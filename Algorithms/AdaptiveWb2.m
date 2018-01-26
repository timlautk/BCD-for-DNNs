function [W2new,b2new,beta5] = AdaptiveWb2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2,W3,V,W2star,b1,b2,b3,b4,b2star,beta,t)
    W2new = W2 + beta*(W2star-W2);
    b2new = b2 + beta*(b2star-b2);
    if loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2star,W3,V,b1,b2star,b3,b4) <= loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2new,W3,V,b1,b2new,b3,b4)
        beta5 = t*beta;
        W2new = W2star;
        b2new = b2star;
    else
%         beta5 = min(beta/t,1);
        beta5 = beta;
    end
    
end