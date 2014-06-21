import operator
import numbers
from . import dcp

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
    
class Expression(object):
    """
    Basic expression class.
    """
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

# Attach basic behavior to Expression object
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
    decorate_operator(dcp.monotonicity(dcp.Affine, dcp.Increasing, dcp.Increasing), x)
decorate_operator(dcp.monotonicity(dcp.Affine, dcp.Increasing, dcp.Decreasing), "sub")
decorate_operator(dcp.monotonicity(dcp.Affine, dcp.Increasing, dcp.Decreasing), "rsub")  # TODO, should be reversed? I don't think so

for x in binary_ops:
    decorate_operator(create_constants, x)
    decorate_operator(create_constants, "r" + x)

