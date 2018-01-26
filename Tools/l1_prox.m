function p = l1_prox(x, lambda)
   p = max(abs(x)-lambda,0).*sign(x); 
end 