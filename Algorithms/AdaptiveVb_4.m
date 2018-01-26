function [W2new,b2new,beta1] = AdaptiveVb_4(lambda,gamma,y,a1,W2,W2star,b2,b2star,beta,t)
    W2new = W2 + beta*(W2star-W2);
    b2new = b2 + beta*(b2star-b2);
    if gamma/2*norm(W2star*a1+b2star-y,'fro')^2+lambda*norm(W2star,'fro')^2 <= gamma/2*norm(W2new*a1+b2new-y,'fro')^2+lambda*norm(W2new,'fro')^2
        beta1 = t*beta;
        W2new = W2star;
        b2new = b2star;
    else
%         beta1 = min(beta/t,1);
        beta1 = beta;
    end

end