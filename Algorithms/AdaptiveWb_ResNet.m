function [W1new,b1new,beta7] = AdaptiveWb_ResNet(lambda,gamma,x,a0,a1,W1,W1star,b1,b1star,beta,t)
    W1new = W1 + beta*(W1star-W1);
    b1new = b1 + beta*(b1star-b1);
    if gamma*norm(W1star*a0+b1star+x-a1,'fro')^2+lambda*norm(W1star,'fro')^2 <= gamma*norm(W1new*a0+b1new+x-a1,'fro')^2+lambda*norm(W1new,'fro')^2
        beta7 = t*beta;
%         W1new = W1star;
%         b1new = b1star;
    else
        beta7 = min(beta/t,1);
%         beta7 = beta;
    end
    
end