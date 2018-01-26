function astar = updatea2(z,u2,u3,W,b,alpha,rho,gamma,act)
    [d,N] = size(u2);
    I = sparse(eye(d));    
    switch act
        case 1 
            hz = max(0,z); % ReLU
        case 2
            hz = tansig(z); % tanh
        case 3
            hz = logsig(z); % sigmoid
    end
    astar = (gamma*(W'*W)+(alpha+rho)*I)\(gamma*W'*(u3-b)+rho*hz+alpha*u2);
    
end