function [W2new,b2new,beta1] = AdaptiveVb_3(lambda,y,a1,W2,W2star,b2,b2star,beta,t)
    W2new = W2 + beta*(W2star-W2);
    b2new = b2 + beta*(b2star-b2);
    if cross_entropy(y,a1,W2star,b2star)+lambda*norm(W2star,'fro')^2 <= cross_entropy(y,a1,W2new,b2new)+lambda*norm(W2new,'fro')^2
        beta1 = t*beta;
%         W2new = W2star;
%         b2new = b2star;
    else
%         beta1 = min(beta/t,1);
        beta1 = beta;
    end

end