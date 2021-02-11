function [A, B, C, Sw, Pinf] = k_quasiperiodic_ss()
% # Construct Quasi-Periodic State Space GP Models

% TODO:
% * determine what the best input type is
% * test of course

    [Aq, Bq, Cq, Swq, Pinfq] = k_somethingelse();
    [Ap, Bp, Cp, Swp, Pinfp] = k_periodic_ss(J);
    
    Nq = size(Aq, 1);
    Np = 2*J+1;
    
    A = kron(Aq, eye(Np)) + kron(eye(Nq), Ap);
    B = kron(Bq, Bp);
    C = kron(Cq, Cp);
    Sw = kron(Swq, Pinfp);
    Pinf = kron(Pinfq, Pinfp);
end
