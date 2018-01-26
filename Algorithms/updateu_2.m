function unew = updateu_2(u,W,b,a1,a2,alpha,gamma)
    unew = (alpha*u-gamma*(W*a1+b-a2))/(gamma+alpha);
end