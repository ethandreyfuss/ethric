class ConstraintExpressions(object):
    def __init__(self, apply_G_func, apply_A_func, apply_GT_func, apply_AT_func):
        self.apply_G_func = apply_G_func
        self.apply_A_func = apply_A_func
        self.apply_GT_func = apply_GT_func
        self.apply_AT_func = apply_AT_func


def solve(constraint_expressions, c, h, b):
    """ Solves a cone constrained convex optimization problem
    via an interior point method.  The specific method used
    is a primal-dual infeasible interior point method.  The
    linear system solves are done via conjugate gradient.

    Will this work?  Let's find out.
    Initial implementation based on:
    http://www.seas.ucla.edu/~vandenbe/ee236a/lectures/hsd.pdf

    minimize    c'*x
    subject to  G*x + s = h
                A*x = b
                s >= 0

    >>> solve(None, None, [0], [0], [0])
    """
    assert isinstance(constraint_expressions, ConstraintExpressions)

    kkt_solver = gen_reduced_kkt_solver(constraint_expressions)

    #TODO: rename c? rename h,b (offsets?)

    ABSTOL = 1e-8
    RELTOL = 1e-8
    FEASTOL = 1e-8
    MINSLACK = 1e-8
    STEP = 0.99
    EXPON = 3

    #x
    point = [0.0] * len(c)
    #y, note that dual is different size from primal slack
    dual_point = [0.0] * len(b)
    #s
    slack = [1.0] * len(h)
    #z
    dual_slack = [1.0] * len(h)

    #introduced as part of extended self-dual embedding
    tau_scale = 1.0
    theta_scale = 1.0
    lambda_slack = 1.0

    #main loop
    MAX_ITERS = 40
    for iter_count in range(MAX_ITERS):
        rx, ry, rz, rt, duality_gap, mu = compute_residuals_and_gap(point, dual_point, slack, dual_slack, constraint_expressions, c, h, b, tau, kappa)
        if evaluate_stopping_criteria():
            break

        compute_affine_scaling_direction()
        select_barrier_parameter()
        compute_search_direction()
        update_iterates()

#tau and kappa are scalars from the extended self-dual embedding
def compute_residuals_and_gap(point, dual_point, slack, dual_slack, constraint_expressions, c, h, b, tau, kappa):
    # /* rx = -A'*y - G'*z - c.*tau */
    rx = [-e1-e2-tau*e3 for e1,e2,e3 in zip(constraint_expressions.apply_AT_func(dual_point),
                                            constraint_expressions.apply_GT_func(dual_slack),
                                            c)]

    # /* ry = A*x - b.*tau */
    ry = [e1-tau*e2 for e1,e2 in zip(constraint_expressions.apply_A_func(point), b)]

    # /* rz = s + G*x - h.*tau */
    rz = [e1+e2-tau*e3 for e1,e2,e3 in zip(slack,
                                           constraint_expressions.apply_G_func(point),
                                           h)]

    rt = kappa + dot(c, point) + dot(b, dual_point) + dot(h, dual_slack)

    duality_gap = dot(slack, dual_slack)
    #mu is barrier cost
    D = num_linear_inequalities = len(h)
    #mu is theta parameter in extended self-dual embedding
    mu = (duality_gap + kappa * tau ) / (num_linear_inequalities + 1)

    return (rx, ry, rz, rt, duality_gap, mu)



def evaluate_stopping_criteria():
    #for now we can just do fixed number of iterations
    return False

def compute_affine_scaling_direction():
    pass

def select_barrier_parameter():
    pass

def compute_search_direction():
    pass

def update_iterates():
    pass

def construct_RHS_affine_scaling_direction(scaling_point, rx, ry, rz, rt, tau, kappa, slack, dual_slack):
    scaling_point = [e1/e2 for zip(slack, dual_slack)]
    ds     = -[e**2 for e in scaling_point]
    dkappa = -(tau * kappa)
    return [rx, ry, rz, rt, ds, dkappa]

def gen_reduced_kkt_solver(apply_G_func, apply_A_func, apply_GT_func, apply_AT_func):
    def kkt_solver(scaling_matrix):
        pass
    return kkt_solver

def dot(vec1, vec2):
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))

def mat_mul_func(mat):
    def op_func(vec):
        rval = []
        vec_len = len(vec)
        for i in range(vec_len):
            rval.append(dot(vec, mat[i*vec_len:(i+1)*vec_len]))
        return rval
    return op_func

def test_IPM():
    def objective(x):
        return 0+x[3]+x[4]+0.5*(0+x[5]+x[6]+x[7]+x[8]+x[9]+0+x[10]+x[11]+x[12]+x[13]+x[14])

    def apply_G(x):
        out = [0.0] * 24
        out[0] = 0-x[3]-x[0]
        out[1] = x[0]-x[3]
        out[2] = 0-x[4]-x[1]
        out[3] = x[1]-x[4]
        out[4] = 1-(0+x[0]*1.36219568593+x[1]*0.788022343248)+x[2]-x[5]
        out[5] = 0-x[5]
        out[6] = 1-(0+x[0]*2.49043044884+x[1]*0.650342412199)+x[2]-x[6]
        out[7] = 0-x[6]
        out[8] = 1-(0+x[0]*-0.165396578471+x[1]*1.9445954487)+x[2]-x[7]
        out[9] = 0-x[7]
        out[10] = 1-(0+x[0]*0.303369894373+x[1]*2.59545656114)+x[2]-x[8]
        out[11] = 0-x[8]
        out[12] = 1-(0+x[0]*1.47375359485+x[1]*1.33955453859)+x[2]-x[9]
        out[13] = 0-x[9]
        out[14] = 1+0+x[0]*3.92210737982+x[1]*4.00472514145-x[2]-x[10]
        out[15] = 0-x[10]
        out[16] = 1+0+x[0]*5.15410131761+x[1]*0.804644500879-x[2]-x[11]
        out[17] = 0-x[11]
        out[18] = 1+0+x[0]*4.09773767354+x[1]*4.09868440499-x[2]-x[12]
        out[19] = 0-x[12]
        out[20] = 1+0+x[0]*3.65777186607+x[1]*3.19434648231-x[2]-x[13]
        out[21] = 0-x[13]
        out[22] = 1+0+x[0]*6.29224931848+x[1]*3.88803256355-x[2]-x[14]
        out[23] = 0-x[14]
        return out

    def apply_GT(x):
        pass

    def apply_A(x):
        return []

    def apply_AT(x):
        pass

def conjugate_gradient(apply_operator_func, target_values, starting_point=None):
    """
    >>> conjugate_gradient(mat_mul_func([1, -1, 2, -1, 5, 8, 2, 8, -4]), [-2, -44, 46])
    [9.0, 0.9999999999999999, -5.000000000000004]
    """
    if starting_point is None:
        starting_point = [0.0] * len(target_values)
    residual = [b-x for x,b in zip(apply_operator_func(starting_point), target_values)]
    last_residual_norm_sq = dot(residual, residual)
    point = starting_point
    last_conjugate_vec = residual
    for k in range(len(target_values)):
        image_of_cg_vec = apply_operator_func(last_conjugate_vec)
        cg_vec_scale = last_residual_norm_sq / dot(last_conjugate_vec, image_of_cg_vec)
        point = [x+cg_vec_scale*y for x, y in zip(point, last_conjugate_vec)]
        residual = [x-cg_vec_scale*y for x, y in zip(residual, image_of_cg_vec)]
        residual_norm_sq = dot(residual, residual)
        #print "residual_norm_sq:",residual_norm_sq
        if residual_norm_sq < 0.000001:
            break
        ratio = residual_norm_sq / last_residual_norm_sq
        last_conjugate_vec = [x+ratio*y for x, y in zip(residual, last_conjugate_vec)]
        last_residual_norm_sq = residual_norm_sq
    #final_img = apply_operator_func(point)
    #print "target_values:",target_values
    #print "final_img:",final_img
    return point

if __name__ == "__main__":
    import doctest
    doctest.testmod()
