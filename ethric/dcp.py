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