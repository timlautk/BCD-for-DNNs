function [Wstar,b4star] = updateWb(y,a,W,b,alpha,gamma,lambda)
    [d,N] = size(a);
    I = sparse(eye(d));
%     alpha1 = gamma*norm(a*a',2);
%     alpha2 = sqrt(d);
    Wstar = (alpha*W+gamma*(y-b)*a')/((alpha+lambda)*I+gamma*(a*a'));
    b4star = (alpha*b+gamma*sum(y-W*a,2))/(gamma*N+alpha);
end