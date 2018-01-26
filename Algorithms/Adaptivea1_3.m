function [a1new,beta6] = Adaptivea1_3(gamma1,gamma2,y,a0,a1,a1star,W1,W2,b1,b2,beta,t)
    a1new = a1 + beta*(a1star-a1);
    if gamma1*norm(W1*a0+b1-a1star,'fro')^2+gamma2*norm(W2*a1star+b2-y,'fro')^2 <= gamma1*norm(W1*a0+b1-a1new,'fro')^2+gamma2*norm(W2*a1new+b2-y,'fro')^2
        beta6 = t*beta;
%         a1new = a1star;
    else
%         beta6 = min(beta/t,1);
        beta6 = beta;
    end

end