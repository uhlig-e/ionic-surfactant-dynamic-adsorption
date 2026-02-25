import numpy as np
from scipy.integrate import solve_bvp
from numerical_solution_object import numerical_solution


def solve(
        x,
        t_f,
        t_0=0,
        dt_0=1e-4,
        eps=1e-2, 
        Dp=1, 
        Dn=1, 
        Da=5, 
        beta=1,
        kap=1, 
        kan=0, 
        Kp=1, 
        Kn=1, 
        zp=1, 
        zn=-1,
        kinetics='Langmuir',
        stepfrac=0.9,
        auto_terminate=True,
        save=True, 
        notes='', 
        filename='', 
        verbose=False) -> object:
    '''
    x: numpy array mesh
    t_f: (float) final dimensionless time to integrate governing equations to
    t_0: (float) initial time (default zero)
    dt_0: initial time step
    eps: ratio of Debye length to adsorption depth
    Dp: positive diffusion coefficient
    Dn: negative diffusion coefficient
    Da: Damkohler number
    beta: dimensionless inverse langmuir adsorption const.
    kap: positive adsorption rate constant
    kan: negative adsorption rate constant
    Kp: positive adsorption equilibrium constant
    Kn: negative adsorption equilibrium constant
    zp: positive signed valence
    zn: negative signed valence
    kinetics: specify either 'Linear' or 'Langmuir' kinetics type
    stepfrac: fraction of the computed (dynamic) time step to take (if zero then takes dt steps every time)
    auto_terminate: boolean to enable the auto-termnination criteria
    save: boolean to save numerical solution object or not
    notes: notes to save for the saved numerical solution object
    filename: file name to save the numerical solution object to
    verbose: print the solver's termination status/message at every time point
    '''
    save_data = locals()

    # initialization
    mesh = x
    initial_guess = np.array(
        [
            np.ones_like(x), 
            np.zeros_like(x), 
            np.ones_like(x), 
            np.zeros_like(x), 
            np.zeros_like(x)              
        ]
    )
    gamma_p_old, gamma_n_old = 0, 0
    forcing = lambda x: np.array([[1], [0], [1], [0], [0]])
    if save: sol_objs = list()

    def bc(X_0, X_infty, p):
        c_p_l, c_p_r = X_0[0], X_infty[0]
        dc_p_l, dc_p_r = X_0[1], X_infty[1]  
        c_n_l, c_n_r = X_0[2], X_infty[2]
        dc_n_l, dc_n_r = X_0[3], X_infty[3]
        dpsi_l, dpsi_r = X_0[4], X_infty[4]
        gamma_p, gamma_n = p

        # left residuals (at interface, x=0)
        if kinetics == 'Linear':
            kinetic_rate_expression_p = Da * (c_p_l - gamma_p)
            kinetic_rate_expression_n = Da * (kan/kap) * (c_n_l - beta * (Kp/Kn) * gamma_n)
        elif kinetics == 'Langmuir':
            kinetic_rate_expression_p = Da * ( c_p_l * (1 - gamma_p) - beta *  gamma_p )
            kinetic_rate_expression_n = Da * (kan/kap) * ( c_n_l * (1 - (Kp/Kn) * gamma_n) - (Kp/Kn) * gamma_n )
        
        gauss = -(eps**(-2)) * (gamma_p - gamma_n) / 2
        R_p_l = (dc_p_l + zp * c_p_l * gauss) - kinetic_rate_expression_p 
        R_n_l = (Dn/Dp) * (dc_n_l + zn * c_n_l * gauss) - kinetic_rate_expression_n
        R_gp = gamma_p - gamma_p_old - dt * (dc_p_l + zp * c_p_l * gauss)
        R_gn = gamma_n - gamma_n_old - dt * (Dn/Dp) * (dc_n_l + zn * c_n_l * gauss) 

        # right residuals (at infinity)
        R_p_r = c_p_r - 1
        R_n_r = c_n_r - 1
        R_dpsi_r = dpsi_r

        # residual 
        residual = np.array([
            R_p_l,
            R_n_l,
            R_dpsi_r,
            R_p_r,
            R_n_r,
            R_gp,
            R_gn
        ])

        return residual
        
    def f(s, X, p):
        '''
        computes RHS of dX/dt = f(t, X), where X = vector of dep. vars
        '''
        c_p = X[0] 
        dc_p = X[1] 
        c_n = X[2] 
        dc_n = X[3] 
        dpsi = X[4] 

        psi_rhs =  -(eps**(-2)) * (zp * c_p + zn * c_n)/2
        d2psi = psi_rhs 
        c_p_rhs = -zp * ( dc_p * dpsi + c_p * d2psi ) + (c_p - forcing(s)[0])/dt
        c_n_rhs = -zn * ( dc_n * dpsi + c_n * d2psi ) + (Dp/Dn) * (c_n - forcing(s)[2])/dt

        X_rhs = np.vstack(
            (
            dc_p,
            c_p_rhs,
            dc_n,
            c_n_rhs,
            psi_rhs
            )
        )

        return X_rhs
    
    t = t_0
    dt = dt_0
    times_as_list = []
    while t + dt < t_f:
        t += dt
        times_as_list.append(t)
        p_vec = [gamma_p_old, gamma_n_old]
        sol = solve_bvp(
            fun=f,
            bc=bc,
            x=mesh,
            y=initial_guess,
            p = p_vec,
            max_nodes=10000000000,
            verbose=verbose
        )
        # time step calculation (if dynamic time-stepping enabled)
        if stepfrac > 0:
            pos_rate = sum( (initial_guess[0] - sol.sol(mesh)[0])**2) / dt**2
            neg_rate = sum( (initial_guess[2] - sol.sol(mesh)[2])**2) / dt**2
            rate = (pos_rate + neg_rate)**(1/2)
            dt = stepfrac/(2 + rate)
                

        # next step initialization
        forcing = sol.sol
        initial_guess = sol.y
        mesh = sol.x
        if gamma_p_old > 0:
            rel_gamma_diff = np.abs(sol.p[0] - gamma_p_old) / gamma_p_old 
        else: 
            rel_gamma_diff = 1
        gamma_p_old, gamma_n_old = sol.p 

        # metrics to display to user

        # print time value
        print(f't = {t:3.3e}, dt = {dt:3.3e}', end='\r') 
        # print solver message if failed or if verbose is switched on
        if verbose or sol.success != 1:
            print(sol)

        if save: sol_objs.append(sol)

        # calculating to see if surf concs aren't changing much
        if auto_terminate and rel_gamma_diff < 1e-6: 
            break

    
    if save:
        obj = numerical_solution(times_as_list, sol_objs, save_data, notes, filename)
        obj.save()

    return obj


