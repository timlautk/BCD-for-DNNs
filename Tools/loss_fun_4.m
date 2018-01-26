function val = loss_fun_4(gamma1,gamma2,gamma3,y,a0,a1,a2,a3,u1,u2,u3,W1,W2,W3,V,b1,b2,b3,b4)
%     [~,N] = size(a0);
%     switch indicator
%         case 1 % ReLU
%             hz1 = max(0,z1); hz2 = max(0,z2); hz3 = max(0,z3); 
%         case 2 % tanh
%             hz1 = tanh_proj(z1); hz2 = tanh_proj(z2); hz3 = tanh_proj(z3);
%         case 3 % sigmoid
%             hz1 = sigmoid_proj(z1); hz2 = sigmoid_proj(z2); hz3 = sigmoid_proj(z3);
%     end
    val = norm(V*a3+b4-y,'fro')^2+gamma1*norm(W1*a0+b1+u1,'fro')^2+gamma2*norm(W2*a1+b2+u2,'fro')^2+gamma3*norm(W3*a2+b3+u3,'fro')^2;
%     val = -1/N*sum(sum(y.*(V*a3+b4-max(V*a3+b4,[],1)-log(sum(exp(V*a3+b4-max(V*a3+b4,[],1)))))))+gamma1*norm(W1*a0+b1-a1,'fro')^2+gamma2*norm(W2*a1+b2-a2,'fro')^2+gamma3*norm(W3*a2+b3-a3,'fro')^2;
    val = val/2;
end