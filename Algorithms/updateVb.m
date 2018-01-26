function [Wstar,bstar] = updateVb(y,a,W,b,alpha,gamma,lambda)
    [K,~] = size(y);
    Wstar = W - 1/alpha*(gamma*(-y*a'+softmax(W*a+b-max(W*a+b,[],2))*a')+lambda*W);
    bstar = b - gamma/alpha*(-ones(K,1)+sum(softmax(W*a+b-max(W*a+b,[],2)),2));
end