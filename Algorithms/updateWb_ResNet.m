function [Wstar,bstar] = updateWb_ResNet(x,y,a,u,W,b,alpha,gamma,lambda)
    [d,N] = size(a);
    I = sparse(eye(d));
%     alpha1 = gamma*norm(a*a',2);
%     alpha2 = sqrt(d);
    Wstar = (alpha*W+gamma*(y-b-x-u)*a')/((alpha+lambda)*I+gamma*(a*a'));
    bstar = (alpha*b+gamma*sum(y-W*a-x-u,2))/(gamma*N+alpha);
end