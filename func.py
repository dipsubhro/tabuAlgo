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

#def schwefel(x):
#   return 418.9829 * len(x) - sum(i * math.sin(math.sqrt(abs(i))) for i in x)

def step(x):
    return sum(int(i)**2 for i in x)

