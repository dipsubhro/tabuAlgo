import statistics
import numpy as np

def tabu_search(func, x0, tenure=2, max_iter=100, bounds=None):
    n=len(x0);x=x0[:];best_x=x[:];best_f=func(x);tabu={};f_values=[]
    for _ in range(max_iter):
        neighbors=[]
        # Generate a fixed number of random neighbors, e.g., 2 * n
        num_random_neighbors = 2 * n # Keep the same number of neighbors as before
        for _ in range(num_random_neighbors):
            x_new = x[:]
            # Choose a random dimension to perturb
            i = np.random.randint(0, n)
            # Generate a random delta within a small range, e.g., -0.5 to 0.5
            # The range of delta should be related to the bounds or problem scale.
            # For now, let's use a fixed small range.
            delta = np.random.uniform(-0.5, 0.5)
            x_new[i] += delta

            if bounds:
                # Ensure the new value stays within bounds
                x_new[i] = max(bounds[0], min(bounds[1], x_new[i]))

            f_new = func(x_new)
            f_values.append(f_new)
            # The 'move' tuple needs to be adjusted as 'd' is no longer -1 or 1
            # I'll use (i, delta) as the move
            neighbors.append(((i,delta),x_new,f_new))
        neighbors.sort(key=lambda t:t[2])
        move=None
        for m,xn,fn in neighbors:
            rev=(m[0],-m[1])
            if rev not in tabu or fn<best_f:
                move,x,fx=m,xn,fn;break
        if move is None:break
        if fx<best_f:best_x,best_f=x[:],fx
        tabu={mv:t-1 for mv,t in tabu.items() if t-1>0}
        tabu[(move[0],-move[1])]=tenure
    
    avg_f = sum(f_values) / len(f_values) if f_values else 0
    median_f = statistics.median(f_values) if f_values else 0
    max_f = max(f_values) if f_values else 0
    
    return best_x, best_f, avg_f, median_f, max_f


