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

def schwefel_222(x):
    """
    Schwefel 2.22 Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum(abs(xi) for xi in x) + math.prod(abs(xi) for xi in x)

def step(x):
    """
    Step Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    return sum(int(i)**2 for i in x)

def rosenbrock(x):
    """
    Rosenbrock Function - Valley-shaped
    Global minimum: f(1, ..., 1) = 0
    Search domain: [-5, 10]
    """
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def rastrigin(x):
    """
    Rastrigin Function - Highly multimodal
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-5.12, 5.12]
    """
    n = len(x)
    return 10*n + sum(i**2 - 10*math.cos(2*math.pi*i) for i in x)

def ackley(x):
    """
    Ackley Function - Multimodal with many local minima
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-32, 32]
    """
    n = len(x)
    s1 = sum(i**2 for i in x)
    s2 = sum(math.cos(2*math.pi*i) for i in x)
    return -20 * math.exp(-0.2 * math.sqrt(s1 / n)) - math.exp(s2 / n) + 20 + math.e

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
    Lévy Function - Widely used in CEC competitions
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

def bent_cigar(x):
    """
    Bent Cigar Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    return x[0]**2 + 1e6 * sum(xi**2 for xi in x[1:])

def high_conditioned_elliptic(x):
    """
    High-Conditioned Elliptic Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    n = len(x)
    return sum((1e6 ** ((i) / (n - 1))) * xi**2 for i, xi in enumerate(x))

def alpine(x):
    """
    Alpine Function (Alpine N.1)
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-10, 10]
    """
    return sum(abs(xi * math.sin(xi) + 0.1 * xi) for xi in x)

def salomon(x):
    """
    Salomon Function
    Global minimum: f(0, ..., 0) = 0
    Search domain: [-100, 100]
    """
    sum_sq = math.sqrt(sum(xi**2 for xi in x))
    return 1 - math.cos(2 * math.pi * sum_sq) + 0.1 * sum_sq
