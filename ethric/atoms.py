from . import expression as expr

def get_problem(args):
    return next((x.problem for x in args if isinstance(x, expr.Expression) and x.problem is not None), None)

def norm1(sequence):
    print "Norm1(",sequence,")"
    return sum(abs(x) for x in sequence)


old_min = min
def min(*args, **kwargs):
    p = get_problem(args)
    if p is not None:
        tmp = p._var()
        for x in args:
            tmp <= x
        return tmp
    else:
        return old_min(*args, **kwargs)


old_max = max
def max(*args, **kwargs):
    p = get_problem(args)
    if p is not None:
        tmp = p._var()
        for x in args:
            tmp >= x
        return tmp
    else:
        return old_max(*args, **kwargs)

def pos(x):
    return max(x, 0)


def neg(x):
    return min(x, 0)



