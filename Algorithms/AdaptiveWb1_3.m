function [W1new,b1new,beta7] = AdaptiveWb1_3(a0,a1,W1,W1star,b1,b1star,beta,t)
    W1new = W1 + beta*(W1star-W1);
    b1new = b1 + beta*(b1star-b1);
    if norm(W1star*a0+b1star-a1,'fro')^2 <= norm(W1new*a0+b1new-a1,'fro')^2
        beta7 = t*beta;
%         W1new = W1star;
%         b1new = b1star;
    else
        beta7 = min(beta/t,1);
%         beta7 = beta;
    end
    
end