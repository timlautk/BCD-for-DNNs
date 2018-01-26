function [a2new,beta4] = Adaptivea2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,a2star,W1,W2,W3,V,b1,b2,b3,b4,beta,t)
    a2new = a2 + beta*(a2star-a2);
    if loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2star,a3,W1,W2,W3,V,b1,b2,b3,b4) <= loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2new,a3,W1,W2,W3,V,b1,b2,b3,b4)
        beta4 = t*beta;
        a2new = a2star;
    else
%         beta4 = min(beta/t,1);
        beta4 = beta;
    end
    
end