function astar = updatea_2(a1,a2,a3,M1,M2,v1,v2,u1,u2,alpha,eta1,eta2,act)
    [d,~] = size(a2);
    I = sparse(eye(d));
%     alpha = norm(eta2*(M2'*M2)+eta1*eye(d));
    astar = (eta2*(M2'*M2)+(alpha+eta1)*I)\(eta2*M2'*(a3-v2-u2)+eta1*(M1*a1+v1+u1)+alpha*a2);
    switch act
        case 0 
            astar = sign(astar);
        case 1 
            astar = max(0,astar); % ReLU
        case 2
            astar = tanh_proj(astar); % tanh
        case 3
            astar = sigmoid_proj(astar); % sigmoid
    end
end