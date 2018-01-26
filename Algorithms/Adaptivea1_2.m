function [a1new,beta6] = Adaptivea1_2(gamma1,y,a0,a1,a1star,W1,W2,b1,b2,beta,t)
    a1new = a1 + beta*(a1star-a1);
    if cross_entropy(y,a1star,W2,b2)+gamma1/2*norm(W1*a0+b1-a1star,'fro')^2 <= cross_entropy(y,a1new,W2,b2)+gamma1/2*norm(W1*a0+b1-a1new,'fro')^2
        beta6 = t*beta;
        a1new = a1star;
    else
%         beta6 = min(beta/t,1);
        beta6 = beta;
    end

end