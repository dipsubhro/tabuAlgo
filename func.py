import math

def sphere(x):
    """
    Sphere Function - Simple unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5.12, 5.12]
    """
    return sum(i**2 for i in x)

def sum_of_squares(x):
    """
    Sum of Squares Function - Unimodal, easy baseline
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum((i + 1) * xi**2 for i, xi in enumerate(x))

def sum_of_different_powers(x):
    """
    Sum of Different Powers Function - Very easy
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(abs(xi) ** (i + 2) for i, xi in enumerate(x))

def step(x):
    """
    Step Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5.12, 5.12]
    """
    return sum(int(i)**2 for i in x)

def brown(x):
    """
    Brown Function - Smooth unimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 4]
    """
    n = len(x)
    return sum((x[i]**2) ** (x[i+1]**2 + 1) + (x[i+1]**2) ** (x[i]**2 + 1) for i in range(n - 1))

def zakharov(x):
    """
    Zakharov Function - Unimodal, bowl-shaped
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5, 10]
    """
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return sum1 + sum2**2 + sum2**4

def dixon_price(x):
    """
    Dixon–Price Function
    Global minimum: f(x*) = 0
    Search domain: [-10, 10]
    """
    n = len(x)
    term1 = (x[0] - 1)**2
    term2 = sum((i + 1) * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n))
    return term1 + term2

def schumer_steiglitz(x):
    """
    Schumer Steiglitz Function - Sum of fourth powers
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum(xi**4 for xi in x)

def csendes(x):
    """
    Csendes Function - Simple smooth function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(xi**6 * (2 + math.sin(1 / (xi + 1e-10))) for xi in x)

def sixth_power(x):
    """
    Sixth Power Function - Smooth, similar to Csendes
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return sum(xi**6 for xi in x)

def powell(x):
    """
    Powell Function - Non-separable
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-4, 5]
    Note: Works best with dims divisible by 4
    """
    n = len(x)
    result = 0
    for i in range(0, n - 3, 4):
        result += (x[i] + 10*x[i+1])**2
        result += 5 * (x[i+2] - x[i+3])**2
        result += (x[i+1] - 2*x[i+2])**4
        result += 10 * (x[i] - x[i+3])**4
    return result

def quartic(x):
    """
    Quartic Function (De Jong's F4)
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1.28, 1.28]
    """
    return sum((i + 1) * xi**4 for i, xi in enumerate(x))

def rotated_hyper_ellipsoid(x):
    """
    Rotated Hyper-Ellipsoid Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-65.536, 65.536]
    """
    n = len(x)
    result = 0
    for i in range(n):
        result += sum(x[j]**2 for j in range(i + 1))
    return result

def discus(x):
    """
    Discus (Tablet) Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    return 1e6 * x[0]**2 + sum(xi**2 for xi in x[1:])

def exponential(x):
    """
    Exponential Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-1, 1]
    """
    return -math.exp(-0.5 * sum(xi**2 for xi in x)) + 1

