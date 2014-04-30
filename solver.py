
def tests():
    """
    >>> tests()
    Testing
    """
    print "Testing"

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
