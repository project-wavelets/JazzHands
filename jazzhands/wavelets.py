import numpy as np
import scipy

__all__ = ['phi_1', 'phi_2', 'phi_3', 'w_alpha', 'N_eff', 'function_inner_product', 'S_matrix', 'inner_product_vector',
           'coeffs', 'y_fit', 'V_x', 'V_y', 'WWZ'] #function names that get imported


"""

WAVELET TRANSFORMS, BASED ON FOSTER 1996

"""

#Defining functions

def phi_1(t,omega,tau):
    """
    Returns 1 for all times; sets the coarse detail in the wavelet transform. Note that 
    the parameters omega and tau have no use here; we only include them to loop over
    the basis functions.
    
    Parameters
    ----------
    t : array-like
        times
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
        
    Returns
    -------
    out : array-like
        array of 1s of length t
    
    """
    return np.ones(len(t))

def phi_2(t,omega,tau):
    """
    Second basis function, cos(omega(t-tau))
    
    Parameters
    ----------
    t : array-like
        times
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
        
    Returns
    -------
    out : array-like
        value of phi_2
    
    """
    return np.cos(omega*(t-tau))

def phi_3(t,omega,tau):
    """
    Third basis function, sin(omega(t-tau))
    
    Parameters
    ----------
    t : array-like
        times of observations
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
        
    Returns
    -------
    out : array-like
        value of phi_3
    
    """
    return np.sin(omega*(t-tau))

def w_alpha(t,omega,tau,c): 
    """
    Weighting function for each point at a given omega and tau; (5-3) in Foster (1996)
    
    Parameters
    ----------
    t : array-like
        times of observations
    omega : float
        angular frequency in radians per unit time. 
    tau : float
        time shift in same units as t
    c : float
        Decay constant of the Gaussian envelope for the wavelet
        
    Returns
    -------
    weights : array-like
        Statistical weights of data points
    
    """
    return np.exp(-c*np.power(omega*(t - tau),2.0))

def N_eff(ws):
    """
    Effective number of points contributing to the transform; (5-4) in Foster (1996)
    
    Parameters
    ----------
    ws : array-like
        weights of observations, already calculated
        
    Returns
    -------
    Neff : float
        Effective number of data points
    
    """
    
    return np.power(np.sum(ws),2.0)/np.sum(np.power(ws,2.0))

def function_inner_product(func1,func2,ws):
    """
    Define the inner product of two functions; (4-2) in Foster (1996)
    
    Parameters
    ----------
    func1 : array-like
        Values of f at times corresponding to the weights
    func2 : array-like
        Values of g at times corresponding to the weights
    ws : array-like
        weights of observations, already calculated
        
    Returns
    -------
    inner_product : float
        Inner product of func1 and func2
    
    """
    num = np.sum(ws*func1*func2)
    den = np.sum(ws)
    return num/den

def S_matrix(func_vals,ws):
    """
    Define the S-matrix; (4-2) in Foster (1996)
    
    Takes the values of the functions already evaluated at the times of observations
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        weights of observations, already calculated
        
    Returns
    -------
    S : `numpy.matrix`
        S-matrix; size len(func_vals)xlen(func_vals)
    
    """
    S = np.array([[function_inner_product(f1,f2,ws) for f1 in func_vals] for f2 in func_vals])
    return np.matrix(S)

def inner_product_vector(func_vals,ws,y):
    """
    Generates a column vector consisting of the inner products between the basis
    functions and the observed data
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    phi_y : `numpy.array`
        Column vector where phi_y_i = phi_i * y
    
    """
    return np.array([[function_inner_product(func,y,ws) for func in func_vals]]).T

def coeffs(func_vals,ws,y):
    """
    Calculate the coefficients of each $\phi$. Adapted from (4-4) in Foster (1996)
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    coeffs : `numpy.array`
        Contains coefficients for each basis function
    
    """
    S_m = S_matrix(func_vals,ws)
    phi_y = inner_product_vector(func_vals,ws,y)
    return np.linalg.solve(S_m,phi_y).T

def V_x(f1_vals,ws,y):
    """
    Calculate the weighted variation of the data. Adapted from (5-9) in Foster (1996)
    
    Parameters
    ----------
    f1_vals : array-like
        Array of values of the first basis function; should be equivalent to
        `numpy.ones(len(y))`
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    vx : float
        Weighted variation of the data
    
    """
    return function_inner_product(y,y,ws) - np.power(function_inner_product(f1_vals,y,ws),2.0)

def y_fit(func_vals,ws,y):
    """
    Calculate the value of the model. 
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    y_f : array-like
        Values of the fit model
    y_a : `numpy.array`
        The coefficients returned by `coeffs`
    
    """
    y_a = coeffs(func_vals,ws,y)
    return y_a.dot(func_vals),y_a

def V_y(func_vals,f1_vals,ws,y):
    """
    Calculate the weighted variation of the model. Adapted from (5-10) in Foster (1996) 
    
    Parameters
    ----------
    func_vals : array-like
        Array of values of basis functions at times corresponding to the weights
        Should have shape (number of basis functions,len(ws))
    f1_vals : array-like
        Array of values of the first basis function; should be equivalent to
        `numpy.ones(len(y))`
    ws : array-like
        Weights of observations, already calculated
    y : array-like
        Observed data
        
    Returns
    -------
    vy : float
        Weighted variation of the model
    y_a :float
        Coefficients from `coeffs`
    
    """
    y_f,y_a = y_fit(func_vals,ws,y)
    return function_inner_product(y_f,y_f,ws) - np.power(function_inner_product(f1_vals,y_f,ws),2.0),y_a

def WWZ(func_list,f1,y,t,omega,tau,c=0.0125,exclude=True):
    """
    Calculate the Weighted Wavelet Transform of the data `y`, measured at times `t`,
    evaluated at a wavelet scale $\omega$ and shift $\tau$, for a decay factor of the
    Gaussian envelope `c`. Adapted from (5-11) in Foster (1996)
    
    Parameters
    ----------
    func_list : array-like
        Array or list containing the basis functions, not yet evaluated
    f1 : array-like
        First basis function. Should be equivalent to `lambda x: numpy.ones(len(x))`
    y : array-like
        Observed data
    t : array-like
        Times of observations
    omega : float
        Scale of wavelet; corresponds to an angular frequency
    tau : float
        Shift of wavelet; corresponds to a time
    c : float
        Decay rate of Gaussian envelope of wavelet. Default 0.0125
    exclude : bool
        If exclude is True, returns 0 if the nearest data point is more than one cycle away.
        Default True.
        
    Returns
    -------
    WWZ : float
        WWZ of the data at the given frequency/time.
    WWA : float
        Corresponding amplitude of the signal at the given frequency/time
    
    """
    
    if exclude and (np.min(np.abs(t-tau)) > 2.0*np.pi/omega):
        return 0.0, 0.0
    
    ws = w_alpha(t,omega,tau,c)
    Neff = N_eff(ws)
    
    func_vals = np.array([f(t,omega,tau) for f in func_list])
    
    f1_vals = f1(t,omega,tau)
    
    Vx = V_x(f1_vals,ws,y)
    Vy,y_a = V_y(func_vals,f1_vals,ws,y)
    
    y_a_rows = y_a[0]
    
    return ((Neff - 3.0) * Vy)/(2.0 * (Vx - Vy)),np.sqrt(np.power(y_a_rows[1],2.0)+np.power(y_a_rows[2],2.0))

def arg_wrapper(args):
    return WWZ(*args)


def MP_WWZ(func_list,f1,y,t,omegas,taus,
           c=0.0125,exclude=True,mp=True,n_processes=None):
    """
    Calculate the Weighted Wavelet Transform of the data `y`, measured at times `t`,
    evaluated on a grid of wavelet scales `omegas` and shifts `taus`, for a decay factor of 
    the Gaussian envelope `c`. Adapted from (5-11) in Foster (1996).
    
    Note that this can be incredibly slow for a large enough light curve and a dense enough 
    grid of omegas and taus, so we include multiprocessing to speed it up.
    
    Parameters
    ----------
    func_list : array-like
        Array or list containing the basis functions, not yet evaluated
    f1 : array-like
        First basis function. Should be equivalent to `lambda x: numpy.ones(len(x))`
    y : array-like
        Observed data
    t : array-like
        Times of observations
    omega : array-like
        Scale of wavelets; corresponds to an angular frequency
    tau : array-like
        Shift of wavelets; corresponds to a time
    c : float
        Decay rate of Gaussian envelope of wavelet. Default 0.0125
    exclude : bool
        If exclude is True, returns 0 if the nearest data point is more than one cycle away.
        Default True.
    mp : bool
        If `mp` is True, uses the `multiprocessing.Pool` object to calculate the WWZ
        at each point. Default True
    n_processes : int
        If `mp` is True, sets the `processes` parameter of `multiprocessing.Pool`. If not given,
        sets to `multiprocessing.cpu_count()-1`
        
    Returns
    -------
    wwz : array-like
        WWZ of the data evaluated on the frequency/time grid. Shape is 
        `(len(omegas),len(taus))`
    wwa : array-like
        WWA of the data evaluated on the frequency/time grid. Shape is 
        `(len(omegas),len(taus))`
    
    """
    if not n_processes:
        n_processes = multiprocessing.cpu_count() - 1
        
    if isnotebook():
        this_tqdm = nb_tqdm
    else:
        this_tqdm = tqdm
        
    if mp:
        args = np.array([[func_list,f1,y,t,omega,tau,c,exclude] for omega in omegas for tau in taus])
        
        """with multiprocessing.Pool(n_processes) as p:
            transform = list(
                        this_tqdm(
                        p.imap(arg_wrapper, *args), total=len(omegas)*len(taus)
                                  )
            )                     
        transform = np.array(transform).reshape(len(omegas),len(taus),2)"""
       
        with multiprocessing.Pool(processes=n_processes) as pool:
            results = pool.starmap(WWZ, args, chunksize=int(len(omegas)*len(taus)/10))
            transform = np.array(results).reshape(len(omegas),len(taus),2)
            wwz = transform[:,:,0]
            wwa = transform[:,:,1]
            
        
    else:
        transform = np.array([[WWZ(func_list,f1,y,t,omega,tau,c,exclude) for tau in taus] for omega in this_tqdm(omegas)])
        wwz = transform[:,:,0].T
        wwa = transform[:,:,1].T
    
    return wwz,wwa
