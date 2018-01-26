function astar = updatea(u1,u2,u3,M1,M2,v1,v2,alpha,eta1,eta2,act)
    [d,N] = size(u2);
    I = sparse(eye(d));
%     alpha = norm(eta2*(M2'*M2)+eta1*eye(d));
    astar = (eta2*(M2'*M2)+(alpha+eta1)*I)\(eta2*M2'*(u3-v2)+eta1*(M1*u1+v1)+alpha*u2);
    switch act
        case 1 
            astar = max(0,astar); % ReLU
        case 2
            astar = tanh_proj(astar); % tanh
        case 3
            astar = sigmoid_proj(astar); % sigmoid
    end
end