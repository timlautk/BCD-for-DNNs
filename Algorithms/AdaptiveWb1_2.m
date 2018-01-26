function [W1new,b1new,beta7] = AdaptiveWb1(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2,W3,V,W1star,b1,b2,b3,b4,b1star,beta,t)
    W1new = W1 + beta*(W1star-W1);
    b1new = b1 + beta*(b1star-b1);
    if loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1star,W2,W3,V,b1star,b2,b3,b4) <= loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1new,W2,W3,V,b1new,b2,b3,b4)
        beta7 = t*beta;
        W1new = W1star;
        b1new = b1star;
    else
%         beta7 = min(beta/t,1);
        beta7 = beta;
    end
    
end