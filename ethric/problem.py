from . import expression as expr

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
        new_var = expr.Variable(self)
        self.temporary_var_list.append(new_var)
        return new_var

    def var(self):
        new_var = expr.Variable(self)
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
        import random
        for v in self.temporary_var_list:
            v.set(random.random())
        print "Objective Val:", self.objective_to_minimize.eval()
        print "   subject to:"
        for constraint in self.less_than_or_equal_zero_constraints:
            print str(constraint.eval()) + " <= 0"
        for constraint in self.equal_zero_constraints:
            print str(constraint.eval()) + " == 0"