function [a1new,beta6] = Adaptivea1(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,a1star,W1,W2,W3,V,b1,b2,b3,b4,beta,t)
    a1new = a1 + beta*(a1star-a1);
    if loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1star,a2,a3,W1,W2,W3,V,b1,b2,b3,b4) <= loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1new,a2,a3,W1,W2,W3,V,b1,b2,b3,b4)
        beta6 = t*beta;
        a1new = a1star;
    else
%         beta6 = min(beta/t,1);
        beta6 = beta;
    end

end