import numbers, operator

#TODO: 
# clean up conversion to Constant
# test cases
# reorganize file

def const_func(function_to_use_on_constants):
    def decorator(func):
        def decorated_func(*args):
            l = [isinstance(arg, Constant) for arg in args]
            all_constants = False not in l
            if all_constants:
                return Constant(function_to_use_on_constants(*[x.val for x in args]))
            else:
                return func(*args)
        return decorated_func
    return decorator

def create_constants(func):
    def decorated_func(self, other):
        if not isinstance(other, Expression):
            other = Constant(other)
        return func(self, other)
    return decorated_func

class Monotonicity(object):
    pass

class Increasing(Monotonicity):
    pass

class Decreasing(Monotonicity):
    pass

class Mixed(Monotonicity):
    pass

class Curvature(object):
    pass

class Convex(Curvature):
    pass

class Concave(Curvature):
    pass

class Affine(Curvature):
    pass

class Unknown(Curvature):
    pass

def is_convex(expr):
    return expr.is_convex or expr.is_affine

def is_concave(expr):
    return expr.is_concave or expr.is_affine

def is_affine(expr):
    return expr.is_affine

def make_convex(expr):
    expr.make_convex()
    
def make_concave(expr):
    expr.make_concave()
    
def make_affine(expr):
    expr.make_affine()

def check_curvature(args, inner_args, increasing_arg_requirement, decreasing_arg_requirement):
    rval = True
    for idx, arg in enumerate(args):
        assert isinstance(arg, Monotonicity), "Invalid monotonicity for monotonicity decorator: "+str(type(arg))
        print "rval",rval,"arg",arg,"inner_arg",inner_args[idx],"cvx",is_convex(inner_args[idx]),"ccv",is_concave(inner_args[idx])
        if isinstance(arg, Increasing):
            rval = rval and increasing_arg_requirement(inner_args[idx])
            #"Convex argument required when function is of increasing"+\
            #" monotonicity, but "+str(arg)+" is not convex"
        elif isinstance(arg, Decreasing):
            rval = rval and decreasing_arg_requirement(inner_args[idx])
            #"Concave argument required when function is of increasing"+\
            #" monotonicity, but "+str(arg)+" is not concave"
        elif isinstance(arg, Mixed):
            rval = rval and is_affine(inner_args[idx])
            #"Affine argument required when function is of mixed"+\
            #" monotonicity, but "+str(arg)+" is not mixed"
    return rval
    
def monotonicity(*args):
    args = list(args)
    for i in range(len(args)):
        t = args[i]
        if isinstance(t, type): #accept type or instance
            args[i] = t()
    function_curvature = args[0]
    output_convex = False
    output_concave = False
    args = args[1:]
        
    def decorator(func):
        def decorated_func(*inner_args, **kwargs):
            if isinstance(function_curvature, Convex) or isinstance(function_curvature, Affine):
                output_convex = check_curvature(args, inner_args, is_convex, is_concave)
            if isinstance(function_curvature, Concave) or isinstance(function_curvature, Affine):
                output_concave = check_curvature(args, inner_args, is_concave, is_convex)

            if output_convex and output_concave:
                output = make_affine
                print str([str(x) for x in inner_args])+" is affine"
            elif output_convex:
                output = make_convex
                print str([str(x) for x in inner_args])+" is convex"
            elif output_concave:
                output = make_concave
                print str([str(x) for x in inner_args])+" is concave"
            else:
                assert False, "Unknown curvature: "+str(args)+" | "+str([str(x) for x in inner_args])
            rval = func(*inner_args, **kwargs)
            output(rval)
            return rval
        return decorated_func
    return decorator

class Expression(object):
    def __init__(self, LHS, RHS, op):
        if LHS.problem is not None:
            self.problem = LHS.problem
        else:
            self.problem = RHS.problem
        self.LHS = LHS
        self.RHS = RHS
        self.operator = op
        self.is_convex = False
        self.is_concave = False
        self.is_affine = False
        
    def __str__(self):
        return "[" + ", ".join(str(x) for x in [self.LHS, self.RHS, self.operator])+"]"
    
    def _var(self):
        assert self.problem is not None
        return self.problem._var()
    
    def __abs__(self):
        tmp = self._var()
        -tmp<=self<=tmp
        return tmp
        
    def __lt__(self, other):
        raise Exception("Strict comparison not allowed, use <=")

    def __gt__(self, other):
        raise Exception("Strict comparison not allowed, use >=")
        
    def __ne__(self, other):
        #TODO could this be supported with epsilons?
        raise Exception("Not equal not supported")
        
    @create_constants
    def __le__(self, other):
        assert self.problem is not None
        self.problem.less_than_or_equal_zero_constraints.append(self-other)
        
    @create_constants
    def __ge__(self, other):
        assert self.problem is not None
        self.problem.less_than_or_equal_zero_constraints.append(other-self)
        
    @create_constants
    def __eq__(self, other):
        assert self.problem is not None
        self.problem.equal_zero_constraints.append(self-other)
        
    def __neg__(self):
        return 0-self
    
    def make_convex(self):
        print "marking",self,"convex"
        self.is_convex = True
        
    def make_concave(self):
        print "marking",self,"concave"
        self.is_concave = True
        
    def make_affine(self):
        print "marking",self,"affine"
        self.is_affine = True
    
def norm1(sequence):
    return sum(abs(x) for x in sequence)
        
def as_private(x):
    return "__"+x+"__"
    
def add_operator(func, name):
    setattr(Expression, as_private(name), func)
    
def decorate_operator(dec, name):
    priv_name = as_private(name)
    func = getattr(Expression, priv_name)
    setattr(Expression, priv_name, dec(func))
    
def add_binary_operator(x):
    @const_func(getattr(operator, x))
    def func(self, other):
        return Expression(self, other, x)
    @const_func(getattr(operator, x))
    def rfunc(self, other):
        return Expression(other, self, x)
    add_operator(func,x)
    add_operator(rfunc,"r"+x)
    
binary_ops = ["add", "sub", "mul", "div"]
for x in binary_ops:
    add_binary_operator(x)

def enforce_linear(func):
    def decorated(*args):
        l = [isinstance(arg, Constant) for arg in args]
        any_constants = True in l
        assert any_constants, "You cannot multiply or divide variables by each other in Linear Programs"+str([str(x) for x in args])
        return func(*args)
    return decorated
    
def make_binary_operator_linear(x):
    fname = as_private(x)
    func = getattr(Expression, fname)
    setattr(Expression, fname, enforce_linear(func))
    
#must be linear, can't multiply/divide variable by variable
for x in ["mul", "div", "rmul", "rdiv"]:
    decorate_operator(enforce_linear, x)


decorate_operator(monotonicity(Affine, Increasing, Increasing), "add")
decorate_operator(monotonicity(Affine, Increasing, Increasing), "radd")
decorate_operator(monotonicity(Affine, Increasing, Decreasing), "sub")
decorate_operator(monotonicity(Affine, Increasing, Decreasing), "rsub")
    
for x in binary_ops:
    decorate_operator(create_constants, x)
    decorate_operator(create_constants, "r"+x)
    
class Variable(Expression):
    idx = 0
    
    def __init__(self, problem):
        self.problem = problem
        self.idx = Variable.idx
        Variable.idx +=1
        self.is_affine = True
        self.is_convex = True
        self.is_concave = True
    
    def __str__(self):
        return "Var"+str(self.idx)

def get_problem(args):
    return next((x.problem for x in args if isinstance(x, Expression) and x.problem is not None), None) 
    
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
    
class Constant(Expression):
    def __init__(self, val):
        assert isinstance(val, numbers.Number), "Constant must be of numeric type"
        self.val = val
        self.problem = None
        self.is_affine = True
        self.is_convex = True
        self.is_concave = True
        
    def __abs__(self):
        return Constant(abs(self.val))
        
    def __str__(self):
        return str(self.val)

class Problem(object):
    def __init__(self):
        self.var_list = []
        self.temporary_var_list = [] #variables introduced in canonicalization
        self.less_than_or_equal_zero_constraints = []
        self.equal_zero_constraints = []
        self.objective_to_minimize = None
    
    #creates temporary var, for use in implementation of atoms
    def _var(self):
        new_var = Variable(self)
        self.temporary_var_list.append(new_var)
        return new_var
    
    def var(self):
        new_var = Variable(self)
        self.var_list.append(new_var)
        return new_var
    
    def minimize(self, expression):
        assert self.objective_to_minimize is None, "Can only minimize one thing, already minimizing "+\
            str(self.objective_to_minimize)
        self.objective_to_minimize = expression
        
    def maximize(self, expression):
        assert self.objective_to_minimize is None, "Can only minimize one thing, already minimizing "+\
            str(self.objective_to_minimize)
        self.objective_to_minimize = -expression
    
    def solve(self):
        print "Minimizing:",self.objective_to_minimize
        print "   subject to:"
        for constraint in self.less_than_or_equal_zero_constraints:
            print str(constraint)+" <= 0"
        for constraint in self.equal_zero_constraints:
            print str(constraint)+" == 0"
         
def tests():
    p = Problem()
    x1 = p.var()
    exp = (x1+3*5)*4
    p.minimize(exp)
    print str(exp)
    exp2 = 5.5*x1
    print str(exp2)
    exp3 = 5.5*Constant(2.5)+3
    print str(exp3)
    exp4 = 2*p.var()*3
    print str(exp4)
    exp5 = sum([p.var(), p.var(), 2.5])
    print exp5
    exp6 = -x1
    print exp6
    y = [p.var() for _ in range(5)]
    exp7 = norm1(y)
    print exp7
    p.var()<=3
    p.solve()
    
def l1SVM():
    import random
    pos_samples = [(random.gauss(1, 1), random.gauss(1,1)) for _ in range(5)]
    neg_samples = [(random.gauss(5, 1), random.gauss(3,1)) for _ in range(5)]
    print pos_samples[:50]
    print neg_samples[:50]
    def dot(vec1, vec2):
        return sum(v1*v2 for v1,v2 in zip(vec1, vec2))
    
    p = Problem()
    a = [p.var() for _ in range(2)]
    b = p.var()
    gamma = 0.5
    p.minimize(norm1(a)+gamma*(sum(pos(1-dot(a, x)+b) for x in pos_samples)+\
                               sum(pos(1+dot(a, y)+b) for y in neg_samples)))
    p.solve()

if __name__ == "__main__":
    #tests()
    l1SVM()