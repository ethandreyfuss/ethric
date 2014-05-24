import numbers
import operator
import random


# TODO:
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


def check_curvature(
        args,
        inner_args,
        increasing_arg_requirement,
        decreasing_arg_requirement):
    rval = True
    for idx, arg in enumerate(args):
        assert isinstance(arg, Monotonicity), "Invalid monotonicity for monotonicity decorator: " + str(type(arg))
        # print "rval",rval,"arg",arg,"inner_arg",inner_args[idx],"cvx",is_convex(inner_args[idx]),"ccv",is_concave(inner_args[idx])
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
        if isinstance(t, type):  # accept type or instance
            args[i] = t()
    function_curvature = args[0]
    args = args[1:]

    def decorator(func):
        def decorated_func(*inner_args, **kwargs):
            output_convex = False
            output_concave = False
            if isinstance(function_curvature, Convex) or isinstance(function_curvature, Affine):
                output_convex = check_curvature(
                    args,
                    inner_args,
                    is_convex,
                    is_concave)
            if isinstance(function_curvature, Concave) or isinstance(function_curvature, Affine):
                output_concave = check_curvature(
                    args,
                    inner_args,
                    is_concave,
                    is_convex)

            if output_convex and output_concave:
                output = make_affine
                # print str([str(x) for x in inner_args])+" is affine"
            elif output_convex:
                output = make_convex
                # print str([str(x) for x in inner_args])+" is convex"
            elif output_concave:
                output = make_concave
                # print str([str(x) for x in inner_args])+" is concave"
            else:
                assert False, "Unknown curvature: " + str(args) + " | " + str([str(x) for x in inner_args])
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
        self.offset = None
        #TODO deal with offsets somehow

    def __str__(self):
        return "[" + ", ".join(str(x) for x in [self.LHS, self.RHS, self.operator]) + "]"

    def adj_str(self, add_parens=False):
        pl = False
        pr = False
        if self.operator == "mul":
            if hasattr(self.LHS, "operator") and self.LHS.operator in ["add", "sub"]:
                pl = True
            if hasattr(self.RHS, "operator") and self.RHS.operator in ["add", "sub"]:
                pr = True
        elif self.operator == "sub":
            pr = True
        rval = str(self.LHS.adj_str(pl))+str(self.op_to_str(self.operator))+str(self.RHS.adj_str(pr))
        if add_parens:
            return "("+rval+")"
        else:
            return rval

    def op_to_str(self, op):
        if op == "add":
            return "+"
        if op == "sub":
            return "-"
        if op == "mul":
            return "*"
        if op == "div":
            return "/"
        return "?"

    def _var(self):
        assert self.problem is not None
        return self.problem._var()

    def __abs__(self):
        tmp = self._var()
        -tmp <= self <= tmp
        return tmp

    def __lt__(self, other):
        raise Exception("Strict comparison not allowed, use <=")

    def __gt__(self, other):
        raise Exception("Strict comparison not allowed, use >=")

    def __ne__(self, other):
        # TODO could this be supported with epsilons?
        raise Exception("Not equal not supported")

    @create_constants
    def __le__(self, other):
        assert self.problem is not None
        self.problem.less_than_or_equal_zero_constraints.append(self - other)
        return True

    @create_constants
    def __ge__(self, other):
        assert self.problem is not None
        self.problem.less_than_or_equal_zero_constraints.append(other - self)
        return True

    @create_constants
    def __eq__(self, other):
        assert self.problem is not None
        self.problem.equal_zero_constraints.append(self - other)
        return True

    def __neg__(self):
        return 0 - self

    def make_convex(self):
        # print "marking",self,"convex"
        self.is_convex = True

    def make_concave(self):
        # print "marking",self,"concave"
        self.is_concave = True

    def make_affine(self):
        # print "marking",self,"affine"
        self.is_affine = True

    def eval(self):
        # print "Evaling:",str(self)
        func = getattr(operator, self.operator)
        return func(self.LHS.eval(), self.RHS.eval())


def norm1(sequence):
    print "Norm1(",sequence,")"
    return sum(abs(x) for x in sequence)


def as_private(x):
    return "__" + x + "__"


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
    add_operator(func, x)
    add_operator(rfunc, "r" + x)

binary_ops = ["add", "sub", "mul", "div"]
for x in binary_ops:
    add_binary_operator(x)


def enforce_linear(func):
    def decorated(*args):
        l = [isinstance(arg, Constant) for arg in args]
        any_constants = True in l
        assert any_constants, "You cannot multiply or divide variables by each other in Linear Programs" + \
            str([str(x) for x in args])
        return func(*args)
    return decorated


def make_binary_operator_linear(x):
    fname = as_private(x)
    func = getattr(Expression, fname)
    setattr(Expression, fname, enforce_linear(func))

# TODO cannot divide constant by variable, needs enforcement
# must be linear, can't multiply/divide variable by variable
for x in ["mul", "div", "rmul", "rdiv"]:
    decorate_operator(enforce_linear, x)

for x in ["mul", "rmul", "add", "radd", "div", "rdiv"]:
    decorate_operator(monotonicity(Affine, Increasing, Increasing), x)
decorate_operator(monotonicity(Affine, Increasing, Decreasing), "sub")
decorate_operator(monotonicity(Affine, Increasing, Decreasing), "rsub")  # TODO, should be reversed? I don't think so

for x in binary_ops:
    decorate_operator(create_constants, x)
    decorate_operator(create_constants, "r" + x)


class Variable(Expression):
    idx = 0

    def __init__(self, problem):
        self.problem = problem
        self.idx = Variable.idx
        Variable.idx += 1
        self.is_affine = True
        self.is_convex = True
        self.is_concave = True
        self.val = None
        self.offset = 0.0

    def __str__(self):
        return "Var" + str(self.idx)

    def adj_str(self, add_parens):
        return "x["+str(self.idx)+"]"

    def set(self, val):
        self.val = val

    def eval(self):
        return self.val


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
        self.offset = val

    def __abs__(self):
        return Constant(abs(self.val))

    def __str__(self):
        return str(self.val)

    def adj_str(self, add_parens):
        return str(self.val)

    def eval(self):
        return self.val


class Problem(object):

    def __init__(self):
        self.var_list = []
        # variables introduced in canonicalization
        self.temporary_var_list = []
        self.less_than_or_equal_zero_constraints = []
        self.equal_zero_constraints = []
        self.objective_to_minimize = None

    # creates temporary var, for use in implementation of atoms
    def _var(self):
        new_var = Variable(self)
        self.temporary_var_list.append(new_var)
        return new_var

    def var(self):
        new_var = Variable(self)
        self.var_list.append(new_var)
        return new_var

    def vars(self, num):
        return [self.var() for _ in range(num)]

    def minimize(self, expression):
        assert self.objective_to_minimize is None, "Can only minimize one thing, already minimizing " +\
            str(self.objective_to_minimize)
        self.objective_to_minimize = expression

    def maximize(self, expression):
        assert self.objective_to_minimize is None, "Can only minimize one thing, already minimizing " +\
            str(self.objective_to_minimize)
        self.objective_to_minimize = -expression

    def solve(self):
        print "Minimizing:", self.objective_to_minimize
        print "   subject to:"
        for constraint in self.less_than_or_equal_zero_constraints:
            print str(constraint) + " <= 0"
        for constraint in self.equal_zero_constraints:
            print str(constraint) + " == 0"

    def print_apply(self):
        print "Minimizing:",self.objective_to_minimize.adj_str(False)
        print "   subject to:"
        print "out = [0.0] * "+str(len(self.less_than_or_equal_zero_constraints))
        for idx, constraint in enumerate(self.less_than_or_equal_zero_constraints):
            print "out["+str(idx)+"] = "+constraint.adj_str(False)
        for constraint in self.equal_zero_constraints:
            print constraint.adj_str(False) + " == 0"

    def print_apply_adjoint(self):
        print "=== Adjoint ==="
        print "out = [0.0] * "+str(len(self.var_list)+len(self.temporary_var_list))
        for idx, constraint in enumerate(self.less_than_or_equal_zero_constraints):
            print "\n".join(constraint.adj_str("out["+str(idx)+"]"))


    def eval(self):
        # TODO: just for testing
        for v in self.temporary_var_list:
            v.set(random.random())
        print "Objective Val:", self.objective_to_minimize.eval()
        print "   subject to:"
        for constraint in self.less_than_or_equal_zero_constraints:
            print str(constraint.eval()) + " <= 0"
        for constraint in self.equal_zero_constraints:
            print str(constraint.eval()) + " == 0"


def set_vars(vars, vals):
    assert len(vars) == len(vals),\
        "Cannot assign vals list to var list of different size. len(vars):" + \
        str(len(vars)) + ", len(vals):" + str(len(vals))
    for idx, x in enumerate(vals):
        vars[idx].set(x)


def tests():
    p = Problem()
    x1 = p.var()
    exp = (x1 + 3 * 5) * 4
    p.minimize(exp)
    print str(exp)
    exp2 = 5.5 * x1
    print str(exp2)
    exp3 = 5.5 * Constant(2.5) + 3
    print str(exp3)
    exp4 = 2 * p.var() * 3
    print str(exp4)
    exp5 = sum([p.var(), p.var(), 2.5])
    print exp5
    exp6 = -x1
    print exp6
    y = [p.var() for _ in range(5)]
    exp7 = norm1(y)
    print exp7
    p.var() <= 3
    p.solve()


def l1SVM():
    pos_samples = [(random.gauss(1, 1), random.gauss(1, 1)) for _ in range(5)]
    neg_samples = [(random.gauss(5, 1), random.gauss(3, 1)) for _ in range(5)]
    print pos_samples[:50]
    print neg_samples[:50]

    def dot(vec1, vec2):
        return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

    p = Problem()
    a = p.vars(2)
    b = p.var()
    gamma = 0.5
    p.minimize(
        norm1(a) + gamma *
        (sum(pos(1 - dot(a, x) + b) for x in pos_samples) +
         sum(pos(1 + dot(a, y) - b) for y in neg_samples))
    )
    #  + gamma *
    #    (sum(pos(1 - dot(a, x) + b) for x in pos_samples) +
    #     sum(pos(1 + dot(a, y) - b) for y in neg_samples))

    p.print_apply()
    #p.print_apply_adjoint()

    set_vars(a, [1, 2])
    b.set(1)
    #p.eval()

def all_tests():
    """Runs some tests
    >>> random.seed(3)
    >>> l1SVM()
    [(1.0947080382873042, 2.2500243810835503), (0.06862163227929297, 1.9923772805192401), (0.7408454623065659, 0.738489016018826), (2.899725278464757, 1.1575370716337192), (0.9570757462077557, 1.729498486435609)]
    [(6.126853962383855, 2.9691567249650697), (5.5879937451803645, 2.0262756425343023), (4.633209530019471, 2.561874965597218), (3.667716835408852, 1.4914858728389275), (3.373087270042972, 2.7613471445561966)]
    Minimizing: [[[0, Var3, add], Var4, add], [0.5, [[[[[[0, Var5, add], Var6, add], Var7, add], Var8, add], Var9, add], [[[[[0, Var10, add], Var11, add], Var12, add], Var13, add], Var14, add], add], mul], add]
       subject to:
    [[0, Var3, sub], Var0, sub] <= 0
    [[0, Var4, sub], Var1, sub] <= 0
    [[[1, [[0, [Var0, 1.09470803829, mul], add], [Var1, 2.25002438108, mul], add], sub], Var2, add], Var5, sub] <= 0
    [0, Var5, sub] <= 0
    [[[1, [[0, [Var0, 0.0686216322793, mul], add], [Var1, 1.99237728052, mul], add], sub], Var2, add], Var6, sub] <= 0
    [0, Var6, sub] <= 0
    [[[1, [[0, [Var0, 0.740845462307, mul], add], [Var1, 0.738489016019, mul], add], sub], Var2, add], Var7, sub] <= 0
    [0, Var7, sub] <= 0
    [[[1, [[0, [Var0, 2.89972527846, mul], add], [Var1, 1.15753707163, mul], add], sub], Var2, add], Var8, sub] <= 0
    [0, Var8, sub] <= 0
    [[[1, [[0, [Var0, 0.957075746208, mul], add], [Var1, 1.72949848644, mul], add], sub], Var2, add], Var9, sub] <= 0
    [0, Var9, sub] <= 0
    [[[1, [[0, [Var0, 6.12685396238, mul], add], [Var1, 2.96915672497, mul], add], add], Var2, add], Var10, sub] <= 0
    [0, Var10, sub] <= 0
    [[[1, [[0, [Var0, 5.58799374518, mul], add], [Var1, 2.02627564253, mul], add], add], Var2, add], Var11, sub] <= 0
    [0, Var11, sub] <= 0
    [[[1, [[0, [Var0, 4.63320953002, mul], add], [Var1, 2.5618749656, mul], add], add], Var2, add], Var12, sub] <= 0
    [0, Var12, sub] <= 0
    [[[1, [[0, [Var0, 3.66771683541, mul], add], [Var1, 1.49148587284, mul], add], add], Var2, add], Var13, sub] <= 0
    [0, Var13, sub] <= 0
    [[[1, [[0, [Var0, 3.37308727004, mul], add], [Var1, 2.76134714456, mul], add], add], Var2, add], Var14, sub] <= 0
    [0, Var14, sub] <= 0
    Objective Val: 3.86181813428
       subject to:
    -1.67141147537 <= 0
    -2.06403143823 <= 0
    -4.35298704674 <= 0
    -0.758230246287 <= 0
    -2.64447577625 <= 0
    -0.591099582931 <= 0
    -0.51909115386 <= 0
    -0.301267659516 <= 0
    -3.2458111732 <= 0
    -0.0310117514697 <= 0
    -3.28159995606 <= 0
    -0.865527236979 <= 0
    13.5924183236 <= 0
    -0.472749088665 <= 0
    10.9217211062 <= 0
    -0.718823924066 <= 0
    10.878146661 <= 0
    -0.878812800255 <= 0
    7.93655909748 <= 0
    -0.714129483611 <= 0
    9.97468289157 <= 0
    -0.921098667584 <= 0

    >>> tests()
    [[Var15, 15, add], 4, mul]
    [5.5, Var15, mul]
    16.75
    [[2, Var16, mul], 3, mul]
    [[[0, Var17, add], Var18, add], 2.5, add]
    [0, Var15, sub]
    [[[[[0, Var24, add], Var25, add], Var26, add], Var27, add], Var28, add]
    Minimizing: [[Var15, 15, add], 4, mul]
       subject to:
    [[0, Var24, sub], Var19, sub] <= 0
    [[0, Var25, sub], Var20, sub] <= 0
    [[0, Var26, sub], Var21, sub] <= 0
    [[0, Var27, sub], Var22, sub] <= 0
    [[0, Var28, sub], Var23, sub] <= 0
    [Var29, 3, sub] <= 0
    """

if __name__ == "__main__":
    #import doctest
    #doctest.testmod()
    l1SVM()
