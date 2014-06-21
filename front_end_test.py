import random
import ethric

# TODO:
# clean up conversion to Constant
# test cases
# reorganize file


def set_vars(vars, vals):
    assert len(vars) == len(vals),\
        "Cannot assign vals list to var list of different size. len(vars):" + \
        str(len(vars)) + ", len(vals):" + str(len(vals))
    for idx, x in enumerate(vals):
        vars[idx].set(x)


def tests():
    p = ethric.Problem()
    x1 = p.var()
    exp = (x1 + 3 * 5) * 4
    p.minimize(exp)
    print str(exp)
    exp2 = 5.5 * x1
    print str(exp2)
    exp3 = 5.5 * ethric.Constant(2.5) + 3
    print str(exp3)
    exp4 = 2 * p.var() * 3
    print str(exp4)
    exp5 = sum([p.var(), p.var(), 2.5])
    print exp5
    exp6 = -x1
    print exp6
    y = [p.var() for _ in range(5)]
    exp7 = ethric.norm1(y)
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

    p = ethric.Problem()
    a = p.vars(2)
    b = p.var()
    gamma = 0.5
    p.minimize(
        ethric.norm1(a) + gamma *
        (sum(ethric.pos(1 - dot(a, x) + b) for x in pos_samples) +
         sum(ethric.pos(1 + dot(a, y) - b) for y in neg_samples))
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
