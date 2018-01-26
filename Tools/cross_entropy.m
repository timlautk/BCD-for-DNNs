function val = cross_entropy(y,x,V,b)
    [~,N] = size(x);
    val = -sum(sum(y.*(V*x+b-max(V*x+b,[],1)-log(sum(exp(V*x+b-max(V*x+b,[],1)))))))/N;
end