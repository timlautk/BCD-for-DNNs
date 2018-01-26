function val = loss_fun(gamma1,gamma2,gamma3,y,U0,U1,U2,U3,W10,W21,W32,V)
    val = norm(y-V*U3,'fro')^2/60000+gamma1*norm(U1-W10*U0,'fro')^2+gamma2*norm(U2-W21*U1-U0,'fro')^2+gamma3*norm(U3-W32*U2-U0-U1,'fro')^2;
    val = val/2;
end