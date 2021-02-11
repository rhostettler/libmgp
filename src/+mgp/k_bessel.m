function k = k_bessel(x1, x2, l_gp, var_gp, nu_gp)

    r = norm(x1-x2);
    n = size(r, 1);
    
    % check such that nu is big enough
    if nu_gp < (n-2)/2
        error('Parameter error');
    end
    
    if r == 0
        % TODO: Is this really true?
        k = var_gp;
    else
        k = var_gp*gamma(nu_gp+1)*2^nu_gp*(2*pi*r/l_gp)^(-nu_gp)*besselj(nu_gp, 2*pi*r/l_gp);
    end
end
