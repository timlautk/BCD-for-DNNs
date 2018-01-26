function [a3new,beta2] = Adaptivea3_3(gamma3,gamma4,y,a2,a3,a3star,W3,V,b3,c,beta,t)
    [~,N] = size(a3);
    a3new = a3 + beta*(a3star-a3);
    if gamma4/2*norm(V*a3star+c-y,'fro')^2+gamma3/2*norm(W3*a2+b3-a3star,'fro')^2 <= norm(V*a3new+c-y,'fro')^2/(2*N)+gamma3/2*norm(W3*a2+b3-a3new,'fro')^2
        beta2 = t*beta;
        a3new = a3star;
    else
%         beta2 = min(beta/t,1);
        beta2 = beta;
    end
    
end