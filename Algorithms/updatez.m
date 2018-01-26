function zstar = updatez(z,a1,a2,W,b,rho,gamma,alpha,act)
    switch act
        case 1 
            hz = max(0,z); % ReLU
            if z > 0
                hpz = 1;
            else
                hpz = 0;
            end
        case 2
            hz = tansig(z); % tanh
            hpz = 1./(cosh(z))^2;
        case 3
            hz = logsig(z); % sigmoid
            hpz = logsig(z).*logsig(-z);
    end
    
    zstar = z - (rho*(hz-a2).*hpz+gamma*(z-W*a1-b))/alpha;
end