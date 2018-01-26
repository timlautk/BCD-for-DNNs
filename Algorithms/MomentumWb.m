function [Wnew,bnew] = MomentumWb(W,W0,b,b0,beta)
    Wnew = W0 + beta*(W-W0);
    bnew = b0 + beta*(b-b0);
end