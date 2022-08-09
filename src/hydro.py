import numpy as np

def theta(h, theta_r, theta_s, n, m, alpha) :
    """Calculates volumetric soil moisture

    Args:
        h (float) : water pressure head
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture
        n (float) : van Genuchten model parameter
        m (float) : van Genuchten model parameter
        alpha (float) : van Genuchten model parameter
    
    Returns:
        theta (float) : volumetric soil moisture
    """
    if h >= 0 :
        return theta_s

    return theta_r + (theta_s - theta_r) * ((1 + np.abs(alpha * h)**n)**(-m))

def S(theta, theta_r, theta_s) :
    """Calculates relative saturation

    Args:
        theta (float) : volumetric soil moisture
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture

    Returns:
        S (float) : relative saturation
    """
    return (theta - theta_r) / (theta_s - theta_r)

def h_(theta, theta_r, theta_s, n, m, alpha) :
    """Calculates water pressure head

    Args:
        theta (float) : volumetric soil moisture
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture
        n (float) : van Genuchten model parameter
        m (float) : van Genuchten model parameter
        alpha (float) : van Genuchten model parameter
    
    Returns:
        h (float) : water pressure head
    """
    s = S(theta, theta_r, theta_s)
    return -((s**(-1 / m) - 1)**(1 / n)) / alpha

def k(h, S, k_s, m) :
    """Calculates hydraulic conductivity

    Args:
        h (float) : water pressure head
        S (float) : relative saturation
        k_s (float) : saturated hydraulic conductivity
        m (float) : van Genuchten model parameter
    
    Returns:
        K (float) : hydraulic conductivity
    """
    if h >= 0 :
        return k_s

    return k_s * np.sqrt(S) * ((1 - (1 - S**(1/m))**m)**2)

def C_(h, theta_r, theta_s, n, m, alpha) :
    """Calculates water capacity (dtheta/dh)

    Args:
        h (float) : water pressure head
        theta_r (float) : residual soil moisture
        theta_s (float) : saturated soil moisture
        n (float) : van Genuchten model parameter
        m (float) : van Genuchten model parameter
        alpha (float) : van Genuchten model parameter

    Returns:
        C (float) : water capacity
    """
    if h >= 0 :
        return 10**(-20)
    
    return 10**(-20) + ((theta_s - theta_r) * m * n * alpha * (np.abs(alpha * h)**(n - 1))) / ((1 + np.abs(alpha * h)**n)**(m + 1))
