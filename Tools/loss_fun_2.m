function val = loss_fun_2(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,W1,W2,W3,V,b1,b2,b3,b4)
    [~,N] = size(a0);
%     val = norm(V*a3+b4-y,'fro')^2+gamma1*norm(W1*a0+b1-a1,'fro')^2+gamma2*norm(W2*a1+b2-a2,'fro')^2+gamma3*norm(W3*a2+b3-a3,'fro')^2;
    val = -1/N*sum(sum(y.*(V*a3+b4-max(V*a3+b4,[],1)-log(sum(exp(V*a3+b4-max(V*a3+b4,[],1)))))))+gamma1/2*norm(W1*a0+b1-a1,'fro')^2+gamma2/2*norm(W2*a1+b2-a2,'fro')^2+gamma3/2*norm(W3*a2+b3-a3,'fro')^2;
%     val = val/2;
end