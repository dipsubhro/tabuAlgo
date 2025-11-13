def tabu_search(func, x0, tenure=2, max_iter=100, bounds=None):
    n=len(x0);x=x0[:];best_x=x[:];best_f=func(x);tabu={}
    for _ in range(max_iter):
        neighbors=[]
        for i in range(n):
            for d in(-1,1):
                x_new=x[:];x_new[i]+=d
                if bounds:
                    if not(bounds[0]<=x_new[i]<=bounds[1]):continue
                neighbors.append(((i,d),x_new,func(x_new)))
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
    return best_x,best_f
