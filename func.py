import math

def sphere(x):
    return sum(i**2 for i in x)

def rastrigin(x):
    n = len(x)
    return 10*n + sum(i**2 - 10*math.cos(2*math.pi*i) for i in x)

def ackley(x):
    n = len(x)
    s1 = sum(i**2 for i in x)
    s2 = sum(math.cos(2*math.pi*i) for i in x)
    return -20 * math.exp(-0.2 * math.sqrt(s1 / n)) - math.exp(s2 / n) + 20 + math.e

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def step(x):
    return sum(int(i)**2 for i in x)


def schwefel(x):
    """
    Schwefel Function - Classic, highly multimodal
    Global minimum: f(420.9687, ..., 420.9687) = 0
    Search domain: [-500, 500]
    """
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)

def griewank(x):
    """
    Griewank Function - Popular multimodal function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-600, 600]
    """
    sum_sq = sum(xi**2 for xi in x)
    prod_cos = 1
    for i, xi in enumerate(x):
        prod_cos *= math.cos(xi / math.sqrt(i + 1))
    return 1 + sum_sq / 4000 - prod_cos

def levy(x):
    """
    Levy Function - Widely used in CEC competitions
    Global minimum: f(1, ..., 1) = 0
    Search domain: [-10, 10]
    """
    n = len(x)
    w = [(xi - 1) / 4 + 1 for xi in x]
    
    term1 = math.sin(math.pi * w[0])**2
    term2 = sum((w[i] - 1)**2 * (1 + 10 * math.sin(math.pi * w[i] + 1)**2) for i in range(n - 1))
    term3 = (w[-1] - 1)**2 * (1 + math.sin(2 * math.pi * w[-1])**2)
    
    return term1 + term2 + term3

def zakharov(x):
    """
    Zakharov Function - Unimodal, bowl-shaped
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5, 10]
    """
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return sum1 + sum2**2 + sum2**4

def bohachevsky(x):
    """
    Bohachevsky Function - Classic CEC function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    result = 0
    for i in range(len(x) - 1):
        result += x[i]**2 + 2*x[i+1]**2 - 0.3*math.cos(3*math.pi*x[i]) - 0.4*math.cos(4*math.pi*x[i+1]) + 0.7
    return result

def schaffer_n2(x):
    """
    Schaffer N.2 Function - Very popular, deceptive
    Global minimum: f(0, 0) = 0
    Search domain: [-100, 100]
    Note: Uses first two dimensions for n-dimensional input
    """
    x1, x2 = x[0], x[1] if len(x) > 1 else 0
    num = math.sin(x1**2 - x2**2)**2 - 0.5
    den = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + num / den

def matyas(x):
    """
    Matyas Function - Simple but effective test
    Global minimum: f(0, 0) = 0
    Search domain: [-10, 10]
    """
    x1, x2 = x[0], x[1] if len(x) > 1 else 0
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

def sum_of_squares(x):
    """
    Sum of Squares Function - Unimodal, easy baseline
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum((i + 1) * xi**2 for i, xi in enumerate(x))

def trid(x):
    """
    Trid Function - Non-separable unimodal
    Global minimum: f* = -n(n+4)(n-1)/6 for optimal x
    Search domain: [-n^2, n^2]
    """
    n = len(x)
    sum1 = sum((xi - 1)**2 for xi in x)
    sum2 = sum(x[i] * x[i-1] for i in range(1, n))
    return sum1 - sum2

def booth(x):
    """
    Booth Function - Classic optimization test
    Global minimum: f(1, 3) = 0
    Search domain: [-10, 10]
    """
    x1, x2 = x[0], x[1] if len(x) > 1 else 0
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
