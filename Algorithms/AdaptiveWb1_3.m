function [W1new,b1new,beta1] = AdaptiveWb1_3(lambda,gamma,a0,a1,W1,W1star,b1,b1star,beta,t)
    W1new = W1 + beta*(W1star-W1);
    b1new = b1 + beta*(b1star-b1);
    if gamma/2*norm(W1star*a0+b1star-a1,'fro')^2+lambda*norm(W1star,'fro')^2 <= gamma/2*norm(W1new*a0+b1new-a1,'fro')^2+lambda*norm(W1new,'fro')^2
        beta1 = t*beta;
        W1new = W1star;
        b1new = b1star;
    else
%         beta1 = min(beta/t,1);
        beta1 = beta;
    end
    
end